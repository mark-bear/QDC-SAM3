"""QDC-SAM3 full: global fused semantics with packed infrared supplementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .backend.fusion import CallableFusion
from .backend.interfaces import FusionBackend, SegmentationBackend
from .packing import (
    CanvasConfig,
    PackingPlan,
    Rect,
    pack_boxes,
    render_canvas,
    score_crop_placements,
)
from .proposals import BoxIndex, DISGConfig, predict_boxes
from .reconstruction import (
    QWCRConfig,
    ReconstructionResult,
    quality_weighted_reprojection,
    validate_qwcr_config,
)


@dataclass(frozen=True)
class Support:
    """Aligned visible/infrared support pair and its semantic mask."""

    visible: np.ndarray
    infrared: np.ndarray
    mask: np.ndarray
    text_prompt: str | None = None


@dataclass(frozen=True)
class QDCConfig:
    """Shared canvas layout for the two modality-matched branches."""

    width: int = 1008
    height: int = 1008
    support_fraction: float = 0.6
    support_position: str = "bottom"
    gutter: int = 2
    fill: int = 0


@dataclass(frozen=True)
class Prediction:
    """Source-coordinate outputs and inspectable evidence from both forwards."""

    mask: np.ndarray
    final_confidence: np.ndarray
    fused_mask: np.ndarray
    fused_confidence: np.ndarray
    infrared_mask: np.ndarray
    infrared_confidence: np.ndarray
    fused_canvas: np.ndarray
    infrared_canvas: np.ndarray
    fused: np.ndarray
    infrared_packed: np.ndarray
    boxes: tuple[BoxIndex, ...]
    packing_plan: PackingPlan
    infrared_d: ReconstructionResult
    support_count: int
    config: QDCConfig
    disg_config: DISGConfig
    qwcr_config: QWCRConfig
    fused_branch_scores: tuple[float, ...]
    infrared_branch_scores: tuple[float, ...]
    fused_fallback_applied: bool
    infrared_fallback_applied: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline": "qdc-sam3-full",
            "fusion_rule": "quality-gated-residual-noisy-or",
            "support_count": self.support_count,
            "configuration": {
                "canvas": asdict(self.config),
                "disg": asdict(self.disg_config),
                "qwcr": asdict(self.qwcr_config),
            },
            "output_mask": {
                "shape": list(self.mask.shape),
                "foreground_pixels": int(self.mask.sum()),
                "fused_foreground_pixels": int(self.fused_mask.sum()),
                "infrared_foreground_pixels": int(self.infrared_mask.sum()),
            },
            "forward_1": {
                "support_modality": "visible",
                "query": ["fused_full"],
                "scores": list(self.fused_branch_scores),
                "fallback_applied": self.fused_fallback_applied,
            },
            "forward_2": {
                "support_modality": "infrared",
                "query": ["infrared_packed"],
                "scores": list(self.infrared_branch_scores),
                "fallback_applied": self.infrared_fallback_applied,
            },
            "algorithm_a": {"boxes": [box.as_dict() for box in self.boxes]},
            "algorithm_b": {
                "content_source": "infrared",
                "quality_source": "fused",
                **self.packing_plan.as_dict(),
            },
            "algorithm_d": {
                "sam3_output": self.qwcr_config.sam3_output,
                "local_strength": self.qwcr_config.local_strength,
                "quality_min": self.qwcr_config.quality_min,
                "pixel_threshold": self.qwcr_config.pixel_threshold,
                "output_threshold": self.qwcr_config.output_threshold,
                "tiles": [x.as_dict() for x in self.infrared_d.tile_evidence],
            },
        }

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "mask": output / "mask.png",
            "final_confidence": output / "final-confidence.npy",
            "final_confidence_preview": output / "final-confidence.png",
            "fused_mask": output / "fused-mask.png",
            "infrared_mask": output / "infrared-mask.png",
            "fused_confidence": output / "fused-confidence.npy",
            "infrared_confidence": output / "infrared-confidence.npy",
            "fused_canvas": output / "fused-canvas.png",
            "infrared_canvas": output / "infrared-canvas.png",
            "fused": output / "fused.png",
            "infrared_packed": output / "infrared-packed.png",
            "metadata": output / "metadata.json",
        }
        Image.fromarray(self.mask.astype(np.uint8) * 255).save(paths["mask"])
        Image.fromarray(self.fused_mask.astype(np.uint8) * 255).save(
            paths["fused_mask"]
        )
        Image.fromarray(self.infrared_mask.astype(np.uint8) * 255).save(
            paths["infrared_mask"]
        )
        for key, value in (
            ("final_confidence", self.final_confidence),
            ("fused_confidence", self.fused_confidence),
            ("infrared_confidence", self.infrared_confidence),
        ):
            np.save(paths[key], value.astype(np.float32))
        Image.fromarray(
            np.clip(self.final_confidence * 255.0, 0, 255).astype(np.uint8)
        ).save(paths["final_confidence_preview"])
        for key, value in (
            ("fused_canvas", self.fused_canvas),
            ("infrared_canvas", self.infrared_canvas),
            ("fused", self.fused),
            ("infrared_packed", self.infrared_packed),
        ):
            Image.fromarray(value).save(paths[key])
        paths["metadata"].write_text(
            json.dumps(self.metadata(), indent=2) + "\n", encoding="utf-8"
        )
        return paths


def supplement_confidence(
    fused_confidence: np.ndarray, infrared_confidence: np.ndarray
) -> np.ndarray:
    """Monotonically add IR evidence without suppressing the fused branch."""

    if fused_confidence.shape != infrared_confidence.shape:
        raise ValueError("fused and infrared confidence maps must align")
    fused = np.clip(fused_confidence.astype(np.float32), 0.0, 1.0)
    infrared = np.clip(infrared_confidence.astype(np.float32), 0.0, 1.0)
    return (fused + (1.0 - fused) * infrared).astype(np.float32)


def _validate_rgb(image: np.ndarray, name: str) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or not image.size
    ):
        raise ValueError(f"{name} must be an RGB uint8 array")


def _validate_config(config: QDCConfig) -> None:
    if config.width < 8 or config.height < 8:
        raise ValueError("canvas dimensions must be at least 8 pixels")
    if not 0.0 < config.support_fraction < 1.0:
        raise ValueError("support_fraction must be between zero and one")
    if config.support_position not in {"top", "bottom"}:
        raise ValueError("support_position must be top or bottom")
    if config.gutter < 0 or not 0 <= config.fill <= 255:
        raise ValueError("invalid gutter or fill")


def _support_and_target(config: QDCConfig) -> tuple[Rect, Rect]:
    support_height = round(config.height * config.support_fraction)
    target_height = config.height - support_height - config.gutter
    if min(support_height, target_height) < 1:
        raise ValueError("shared canvas layout has no usable target area")
    if config.support_position == "top":
        return (
            Rect(0, 0, config.width, support_height),
            Rect(0, support_height + config.gutter, config.width, config.height),
        )
    return (
        Rect(0, target_height + config.gutter, config.width, config.height),
        Rect(0, 0, config.width, target_height),
    )


def _support_grid(panel: Rect, count: int, gutter: int) -> tuple[Rect, ...]:
    """Tile support examples inside the existing support region."""

    if count < 1:
        raise ValueError("at least one support is required")
    if count == 1:
        return (panel,)
    columns = int(np.ceil(np.sqrt(count * panel.width / panel.height)))
    columns = max(1, min(count, columns))
    rows = int(np.ceil(count / columns))
    usable_width = panel.width - gutter * (columns - 1)
    usable_height = panel.height - gutter * (rows - 1)
    if usable_width < columns or usable_height < rows:
        raise ValueError("support area is too small for the requested support grid")
    widths = [usable_width // columns] * columns
    heights = [usable_height // rows] * rows
    for index in range(usable_width % columns):
        widths[index] += 1
    for index in range(usable_height % rows):
        heights[index] += 1
    x_offsets = [panel.x0]
    y_offsets = [panel.y0]
    for width in widths[:-1]:
        x_offsets.append(x_offsets[-1] + width + gutter)
    for height in heights[:-1]:
        y_offsets.append(y_offsets[-1] + height + gutter)
    return tuple(
        Rect(
            x_offsets[index % columns],
            y_offsets[index // columns],
            x_offsets[index % columns] + widths[index % columns],
            y_offsets[index // columns] + heights[index // columns],
        )
        for index in range(count)
    )


def _resize(image: np.ndarray, box: Rect) -> np.ndarray:
    return cv2.resize(image, (box.width, box.height), interpolation=cv2.INTER_LINEAR)


def _mask_bounds(mask: np.ndarray) -> Rect:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        raise ValueError("support mask must contain foreground pixels")
    return Rect(
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _prompt_box(mask: np.ndarray, source_shape: tuple[int, int], panel: Rect) -> Rect:
    source = _mask_bounds(mask)
    source_height, source_width = source_shape
    scale_x = panel.width / source_width
    scale_y = panel.height / source_height
    x0 = panel.x0 + min(panel.width - 1, int(np.floor(source.x0 * scale_x)))
    y0 = panel.y0 + min(panel.height - 1, int(np.floor(source.y0 * scale_y)))
    return Rect(
        x0,
        y0,
        max(x0 + 1, min(panel.x1, panel.x0 + int(np.ceil(source.x1 * scale_x)))),
        max(y0 + 1, min(panel.y1, panel.y0 + int(np.ceil(source.y1 * scale_y)))),
    )


class QDCSAM3:
    """Run fused-global and modality-matched IR supplementary forwards."""

    def __init__(
        self,
        segmentation: SegmentationBackend,
        *,
        fusion: FusionBackend,
        disg_config: DISGConfig | None = None,
        config: QDCConfig | None = None,
        qwcr_config: QWCRConfig | None = None,
    ) -> None:
        disg_config = DISGConfig() if disg_config is None else disg_config
        config = QDCConfig() if config is None else config
        qwcr_config = QWCRConfig() if qwcr_config is None else qwcr_config
        _validate_config(config)
        validate_qwcr_config(qwcr_config)
        self.segmentation = segmentation
        self.fusion = CallableFusion(fusion)
        self.disg_config = disg_config
        self.config = config
        self.qwcr_config = qwcr_config

    def _forward(self, canvas, supports, text_prompt, query_panels):
        boxes = tuple(
            _prompt_box(mask, shape, panel) for panel, mask, shape in supports
        )
        evidence = self.segmentation.predict(
            canvas, boxes, text_prompt, query_panels, self.qwcr_config
        )
        if (
            evidence.confidence.shape != canvas.shape[:2]
            or evidence.mask.shape != canvas.shape[:2]
        ):
            raise ValueError(
                "segmentation backend must return canvas-coordinate evidence"
            )
        if (
            not np.isfinite(evidence.confidence).all()
            or np.any(evidence.confidence < 0)
            or np.any(evidence.confidence > 1)
        ):
            raise ValueError("segmentation confidence must be finite and in [0, 1]")
        return evidence

    def __call__(
        self,
        visible: np.ndarray,
        infrared: np.ndarray,
        support: Support,
    ) -> Prediction:
        return self._predict(visible, infrared, (support,))

    def predict_multi_support_canvas(
        self,
        visible: np.ndarray,
        infrared: np.ndarray,
        supports: Sequence[Support],
    ) -> Prediction:
        """Run one dual-forward prediction with all supports on each canvas."""

        return self._predict(visible, infrared, tuple(supports))

    def _predict(
        self,
        visible: np.ndarray,
        infrared: np.ndarray,
        supports: tuple[Support, ...],
    ) -> Prediction:
        if not supports:
            raise ValueError("at least one support is required")
        _validate_rgb(visible, "visible image")
        if infrared.dtype != np.uint8 or infrared.ndim != 2:
            raise ValueError("infrared image must be grayscale uint8")
        if visible.shape[:2] != infrared.shape:
            raise ValueError("visible and infrared images must align")
        for support in supports:
            _validate_rgb(support.visible, "visible support")
            if support.infrared.dtype != np.uint8 or support.infrared.ndim != 2:
                raise ValueError("infrared support must be grayscale uint8")
            if support.visible.shape[:2] != support.infrared.shape:
                raise ValueError("visible and infrared support images must align")
            if support.mask.shape != support.infrared.shape:
                raise ValueError("support mask must align with the support pair")
            _mask_bounds(support.mask)
        prompts = {support.text_prompt for support in supports}
        if len(prompts) > 1:
            raise ValueError("all supports must use the same text prompt")
        text_prompt = supports[0].text_prompt

        boxes = tuple(predict_boxes(infrared, self.disg_config))
        fused = self.fusion(visible, infrared)
        _validate_rgb(fused, "fused image")
        if fused.shape != visible.shape:
            raise ValueError("fusion backend must preserve query dimensions")
        support_region, target = _support_and_target(self.config)
        support_panels = _support_grid(
            support_region, len(supports), self.config.gutter
        )
        packed_panel = target

        plan = pack_boxes(
            boxes,
            infrared.shape,
            CanvasConfig(
                width=packed_panel.width,
                height=packed_panel.height,
                gutter=self.config.gutter,
                gutter_fill=self.config.fill,
            ),
        )
        plan = score_crop_placements(fused, boxes, plan)
        infrared_rgb = cv2.cvtColor(infrared, cv2.COLOR_GRAY2RGB)
        infrared_packed = render_canvas(infrared_rgb, plan, self.config.fill)

        fused_canvas = np.full(
            (self.config.height, self.config.width, 3),
            self.config.fill,
            dtype=np.uint8,
        )
        for support, panel in zip(supports, support_panels, strict=True):
            fused_canvas[panel.y0 : panel.y1, panel.x0 : panel.x1] = _resize(
                support.visible, panel
            )
        fused_canvas[target.y0 : target.y1] = _resize(fused, target)
        fused_branch = self._forward(
            fused_canvas,
            tuple(
                (panel, support.mask, support.visible.shape[:2])
                for support, panel in zip(supports, support_panels, strict=True)
            ),
            text_prompt,
            (target,),
        )
        fused_panel_confidence = fused_branch.confidence[
            target.y0 : target.y1, target.x0 : target.x1
        ]
        fused_confidence = cv2.resize(
            fused_panel_confidence,
            (infrared.shape[1], infrared.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        fused_mask = fused_confidence >= self.qwcr_config.output_threshold

        infrared_canvas = np.full_like(fused_canvas, self.config.fill)
        for support, panel in zip(supports, support_panels, strict=True):
            infrared_support_rgb = cv2.cvtColor(support.infrared, cv2.COLOR_GRAY2RGB)
            infrared_canvas[panel.y0 : panel.y1, panel.x0 : panel.x1] = _resize(
                infrared_support_rgb, panel
            )
        infrared_canvas[
            packed_panel.y0 : packed_panel.y1, packed_panel.x0 : packed_panel.x1
        ] = infrared_packed
        query_panels = (packed_panel,)
        infrared_branch = self._forward(
            infrared_canvas,
            tuple(
                (panel, support.mask, support.infrared.shape)
                for support, panel in zip(supports, support_panels, strict=True)
            ),
            text_prompt,
            query_panels,
        )
        packed_confidence = infrared_branch.confidence[
            packed_panel.y0 : packed_panel.y1, packed_panel.x0 : packed_panel.x1
        ]
        packed_mask = infrared_branch.mask[
            packed_panel.y0 : packed_panel.y1, packed_panel.x0 : packed_panel.x1
        ]
        infrared_d = quality_weighted_reprojection(
            np.zeros(infrared.shape, dtype=np.float32),
            packed_confidence,
            packed_mask,
            plan.placements,
            infrared.shape,
            self.qwcr_config,
        )
        infrared_confidence = infrared_d.final_confidence
        final_confidence = supplement_confidence(fused_confidence, infrared_confidence)
        mask = final_confidence >= self.qwcr_config.output_threshold
        return Prediction(
            mask=mask,
            final_confidence=final_confidence,
            fused_mask=fused_mask,
            fused_confidence=fused_confidence,
            infrared_mask=infrared_d.mask,
            infrared_confidence=infrared_confidence,
            fused_canvas=fused_canvas,
            infrared_canvas=infrared_canvas,
            fused=fused,
            infrared_packed=infrared_packed,
            boxes=boxes,
            packing_plan=plan,
            infrared_d=infrared_d,
            support_count=len(supports),
            config=self.config,
            disg_config=self.disg_config,
            qwcr_config=self.qwcr_config,
            fused_branch_scores=fused_branch.scores,
            infrared_branch_scores=infrared_branch.scores,
            fused_fallback_applied=fused_branch.fallback_applied,
            infrared_fallback_applied=infrared_branch.fallback_applied,
        )
