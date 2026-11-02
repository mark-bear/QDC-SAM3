"""Explicit, local-checkpoint SAM3 image inference adapter."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..packing import Rect
from ..reconstruction import QWCRConfig
from .interfaces import SegmentationEvidence


def _instance_maps(values: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    if values is None:
        return np.empty((0, *shape), dtype=np.float32)
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    maps = np.asarray(values)
    if not maps.size:
        return np.empty((0, *shape), dtype=np.float32)
    if maps.ndim == 4:
        if maps.shape[1] != 1:
            raise ValueError(f"SAM3 {name} must have one mask channel")
        maps = maps[:, 0]
    if maps.ndim == 2:
        maps = maps[None]
    if maps.ndim != 3 or maps.shape[1:] != shape:
        raise ValueError(f"SAM3 {name} do not match the shared canvas")
    return maps.astype(np.float32, copy=False)


def _scores(state: dict[str, Any], count: int) -> np.ndarray:
    values = state.get("scores")
    if values is None:
        return np.ones(count, dtype=np.float32)
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if scores.size != count:
        raise ValueError("SAM3 scores and masks must have matching instance counts")
    return np.clip(scores, 0.0, 1.0)


def extract_sam3_evidence(
    state: dict[str, Any],
    shape: tuple[int, int],
    *,
    mode: str,
    instance_threshold: float | None,
    pixel_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an aggregated confidence map and the corresponding binary union."""

    binary_instances = _instance_maps(state.get("masks"), shape, "masks") > 0
    probability_instances = _instance_maps(
        state.get("masks_logits"), shape, "mask probabilities"
    )
    if probability_instances.shape[0] == 0:
        probability_instances = binary_instances.astype(np.float32)
    if binary_instances.shape[0] == 0:
        binary_instances = probability_instances >= pixel_threshold
    if probability_instances.shape[0] != binary_instances.shape[0]:
        raise ValueError("SAM3 binary masks and probabilities are not aligned")
    if not probability_instances.shape[0]:
        empty_float = np.zeros(shape, dtype=np.float32)
        return empty_float, np.zeros(shape, dtype=bool)

    binary = np.any(binary_instances, axis=0)
    if mode == "binary":
        return binary.astype(np.float32), binary
    if mode != "confidence":
        raise ValueError("SAM3 evidence mode must be binary or confidence")

    scores = _scores(state, probability_instances.shape[0])
    threshold = 0.0 if instance_threshold is None else instance_threshold
    if threshold >= 1.0:
        score_reliability = np.ones_like(scores)
    else:
        retained_position = np.clip((scores - threshold) / (1.0 - threshold), 0, 1)
        score_reliability = 0.5 + 0.5 * retained_position
    confidence = np.max(
        np.clip(probability_instances, 0.0, 1.0) * score_reliability[:, None, None],
        axis=0,
    )
    return confidence.astype(np.float32, copy=False), binary


def _branch_scores(state: dict[str, Any]) -> tuple[float, ...]:
    values = state.get("scores", ())
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


class SAM3Backend:
    def __init__(
        self, processor: Any, *, fallback_confidence_threshold: float | None = 0.1
    ):
        self.processor = processor
        self.fallback_confidence_threshold = fallback_confidence_threshold

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        bpe_path: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        fallback_confidence_threshold: float | None = 0.1,
    ) -> SAM3Backend:
        for path in (checkpoint_path, bpe_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=str(bpe_path),
            device=device,
            eval_mode=True,
            load_from_HF=False,
        )
        return cls(
            Sam3Processor(
                model,
                device=device,
                confidence_threshold=confidence_threshold,
            ),
            fallback_confidence_threshold=fallback_confidence_threshold,
        )

    def predict(
        self,
        canvas: np.ndarray,
        support_boxes: tuple[Rect, ...],
        text_prompt: str | None,
        query_panels: tuple[Rect, ...],
        evidence_config: QWCRConfig,
    ) -> SegmentationEvidence:
        import torch

        device = str(getattr(self.processor, "device", "cpu"))
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else nullcontext()
        )
        # SAM3's image backbone emits BF16 activations and its official image
        # examples keep the complete processor interaction inside this context.
        with torch.inference_mode(), amp:
            state = self.processor.set_image(Image.fromarray(canvas))
            self.processor.reset_all_prompts(state)
            if text_prompt:
                state = self.processor.set_text_prompt(text_prompt, state)
            for prompt in support_boxes:
                normalized = [
                    (prompt.x0 + prompt.x1) / (2 * canvas.shape[1]),
                    (prompt.y0 + prompt.y1) / (2 * canvas.shape[0]),
                    prompt.width / canvas.shape[1],
                    prompt.height / canvas.shape[0],
                ]
                state = self.processor.add_geometric_prompt(
                    state=state, box=normalized, label=True
                )
            initial = getattr(self.processor, "confidence_threshold", None)
            effective = initial
            confidence, mask = extract_sam3_evidence(
                state,
                (canvas.shape[0], canvas.shape[1]),
                mode=evidence_config.sam3_output,
                instance_threshold=effective,
                pixel_threshold=evidence_config.pixel_threshold,
            )
            has_query = any(mask[p.y0 : p.y1, p.x0 : p.x1].any() for p in query_panels)
            fallback = False
            if (
                not has_query
                and self.fallback_confidence_threshold is not None
                and initial is not None
                and self.fallback_confidence_threshold < initial
                and hasattr(self.processor, "set_confidence_threshold")
            ):
                try:
                    state = self.processor.set_confidence_threshold(
                        self.fallback_confidence_threshold, state
                    )
                    effective = self.fallback_confidence_threshold
                    fallback = True
                    confidence, mask = extract_sam3_evidence(
                        state,
                        (canvas.shape[0], canvas.shape[1]),
                        mode=evidence_config.sam3_output,
                        instance_threshold=effective,
                        pixel_threshold=evidence_config.pixel_threshold,
                    )
                finally:
                    self.processor.confidence_threshold = initial
        return SegmentationEvidence(
            confidence, mask, _branch_scores(state), initial, effective, fallback
        )
