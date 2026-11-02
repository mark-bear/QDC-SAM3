"""QIRP: quality-scored infrared region packing and coordinate transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

import cv2
import numpy as np
import rectpack

from .proposals import BoxIndex


@dataclass(frozen=True, order=True)
class Rect:
    """Top-left-origin, half-open integer rectangle."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def intersects(self, other: Rect) -> bool:
        return (
            self.x0 < other.x1
            and other.x0 < self.x1
            and self.y0 < other.y1
            and other.y0 < self.y1
        )

    def as_dict(self) -> dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class CanvasConfig:
    width: int | None = None
    height: int | None = None
    gutter: int = 2
    gutter_fill: int = 127
    scales: tuple[float, ...] = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25)

    @classmethod
    def sam3(cls, **overrides: Any) -> CanvasConfig:
        """Return a square configuration matching SAM3's fixed image input."""

        return cls(width=1008, height=1008, **overrides)


@dataclass(frozen=True)
class CropQuality:
    """Saved A/B quality evidence for one rendered fused-image crop."""

    score: float
    a_score: float
    ir_contrast: float
    fused_contrast: float
    sharpness: float
    entropy: float
    scale_retention: float
    proposal_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "algorithm": "fused-crop-quality-v1",
            "score": self.score,
            "a_score": self.a_score,
            "ir_contrast": self.ir_contrast,
            "fused_contrast": self.fused_contrast,
            "sharpness": self.sharpness,
            "entropy": self.entropy,
            "scale_retention": self.scale_retention,
            "proposal_count": self.proposal_count,
        }


@dataclass(frozen=True)
class TilePlacement:
    index: int
    id: str
    source_ids: tuple[str, ...]
    source_box: Rect
    reserved_box: Rect
    content_box: Rect
    scale_x: float
    scale_y: float
    quality: CropQuality | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "source_ids": list(self.source_ids),
            "source_box": self.source_box.as_dict(),
            "reserved_box": self.reserved_box.as_dict(),
            "content_box": self.content_box.as_dict(),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "rotation": False,
            "quality": None if self.quality is None else self.quality.as_dict(),
        }


@dataclass(frozen=True)
class PackingPlan:
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    gutter: int
    scale: float | None
    placements: tuple[TilePlacement, ...]
    dropped_indices: tuple[int, ...]
    dropped_ids: tuple[str, ...]
    attempts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": "MaxRectsBssf",
            "source_size": [self.source_width, self.source_height],
            "canvas_size": [self.canvas_width, self.canvas_height],
            "coordinate_origin": "top_left",
            "rotation": False,
            "gutter": self.gutter,
            "scale": self.scale,
            "placements": [placement.as_dict() for placement in self.placements],
            "dropped_indices": list(self.dropped_indices),
            "dropped_ids": list(self.dropped_ids),
            "attempts": self.attempts,
        }


_DEFAULT_CONFIG = CanvasConfig()


def _validate_config(config: CanvasConfig) -> None:
    if not isinstance(config, CanvasConfig):
        raise TypeError("config must be a CanvasConfig")
    if (
        (config.width is not None and config.width < 1)
        or (config.height is not None and config.height < 1)
        or config.gutter < 0
    ):
        raise ValueError("canvas dimensions must be positive and gutter non-negative")
    if not 0 <= config.gutter_fill <= 255:
        raise ValueError("gutter_fill must be in [0, 255]")
    if not config.scales or any(scale <= 0 for scale in config.scales):
        raise ValueError("at least one positive packing scale is required")


def _resolve_config(
    config: CanvasConfig, source_shape: tuple[int, int]
) -> CanvasConfig:
    """Fill unspecified canvas dimensions from the aligned input size."""

    _validate_config(config)
    source_height, source_width = source_shape
    return CanvasConfig(
        width=source_width if config.width is None else config.width,
        height=source_height if config.height is None else config.height,
        gutter=config.gutter,
        gutter_fill=config.gutter_fill,
        scales=config.scales,
    )


def _rect(box: BoxIndex) -> Rect:
    return Rect(box.x0, box.y0, box.x1, box.y1)


def _validate_boxes(boxes: Sequence[BoxIndex], source_shape: tuple[int, int]) -> None:
    height, width = source_shape
    if [box.index for box in boxes] != list(range(len(boxes))):
        raise ValueError(
            "Algorithm A box indices must be contiguous and ordered from zero"
        )
    if len({box.id for box in boxes}) != len(boxes):
        raise ValueError("Algorithm A box IDs must be unique")
    for box in boxes:
        if not (0 <= box.x0 < box.x1 <= width and 0 <= box.y0 < box.y1 <= height):
            raise ValueError(f"source box is out of bounds: {box.id}")


def _dimensions(box: BoxIndex, scale: float, gutter: int) -> tuple[int, int, int, int]:
    source = _rect(box)
    content_width = max(1, round(source.width * scale))
    content_height = max(1, round(source.height * scale))
    return (
        content_width,
        content_height,
        content_width + 2 * gutter,
        content_height + 2 * gutter,
    )


def _try_pack(
    boxes: Sequence[BoxIndex], scale: float, config: CanvasConfig
) -> dict[int, tuple[int, int, int, int]] | None:
    ordered = sorted(
        boxes,
        key=lambda box: (
            -_dimensions(box, scale, config.gutter)[2]
            * _dimensions(box, scale, config.gutter)[3],
            box.index,
        ),
    )
    packer = rectpack.newPacker(
        pack_algo=rectpack.MaxRectsBssf,
        sort_algo=rectpack.SORT_NONE,
        rotation=False,
    )
    for box in ordered:
        _, _, width, height = _dimensions(box, scale, config.gutter)
        packer.add_rect(width, height, rid=box.index)
    packer.add_bin(config.width, config.height, count=1)
    packer.pack()
    rectangles = packer.rect_list()
    if len(rectangles) != len(boxes):
        return None
    result = {
        index: (x, config.height - y - height, width, height)
        for _, x, y, width, height, index in rectangles
    }
    return result if len(result) == len(boxes) else None


def _validate_placements(
    placements: Sequence[TilePlacement], config: CanvasConfig
) -> None:
    for placement in placements:
        reserved, content = placement.reserved_box, placement.content_box
        if not (
            0 <= reserved.x0 < reserved.x1 <= config.width
            and 0 <= reserved.y0 < reserved.y1 <= config.height
        ):
            raise RuntimeError(f"out-of-bounds placement: {placement.id}")
        if not (
            reserved.x0 <= content.x0 < content.x1 <= reserved.x1
            and reserved.y0 <= content.y0 < content.y1 <= reserved.y1
        ):
            raise RuntimeError(f"invalid content rectangle: {placement.id}")
    for first, second in combinations(placements, 2):
        if first.reserved_box.intersects(second.reserved_box):
            raise RuntimeError(f"overlapping placements: {first.id}, {second.id}")


def pack_boxes(
    boxes: Sequence[BoxIndex],
    source_shape: tuple[int, int],
    config: CanvasConfig = _DEFAULT_CONFIG,
) -> PackingPlan:
    """Pack A's ordered boxes; drop the largest index last-resort."""

    config = _resolve_config(config, source_shape)
    assert config.width is not None and config.height is not None
    boxes = tuple(boxes)
    _validate_boxes(boxes, source_shape)
    source_height, source_width = source_shape
    if not boxes:
        return PackingPlan(
            source_width,
            source_height,
            config.width,
            config.height,
            config.gutter,
            None,
            (),
            (),
            (),
            0,
        )
    remaining, dropped, attempts = list(boxes), [], 0
    while remaining:
        for scale in config.scales:
            attempts += 1
            packed = _try_pack(remaining, scale, config)
            if packed is None:
                continue
            placements = []
            for box in remaining:
                x, y, reserved_width, reserved_height = packed[box.index]
                content_width, content_height, _, _ = _dimensions(
                    box, scale, config.gutter
                )
                source = _rect(box)
                reserved = Rect(x, y, x + reserved_width, y + reserved_height)
                content = Rect(
                    x + config.gutter,
                    y + config.gutter,
                    x + config.gutter + content_width,
                    y + config.gutter + content_height,
                )
                placements.append(
                    TilePlacement(
                        box.index,
                        box.id,
                        box.source_ids,
                        source,
                        reserved,
                        content,
                        content_width / source.width,
                        content_height / source.height,
                    )
                )
            _validate_placements(placements, config)
            return PackingPlan(
                source_width,
                source_height,
                config.width,
                config.height,
                config.gutter,
                scale,
                tuple(placements),
                tuple(box.index for box in dropped),
                tuple(box.id for box in dropped),
                attempts,
            )
        removal = max(remaining, key=lambda box: box.index)
        remaining.remove(removal)
        dropped.append(removal)
    return PackingPlan(
        source_width,
        source_height,
        config.width,
        config.height,
        config.gutter,
        None,
        (),
        tuple(box.index for box in dropped),
        tuple(box.id for box in dropped),
        attempts,
    )


def render_canvas(
    fused_rgb: np.ndarray, plan: PackingPlan, gutter_fill: int = 127
) -> np.ndarray:
    """Cut fused pixels using source boxes and render them into the plan."""

    if (
        not isinstance(fused_rgb, np.ndarray)
        or fused_rgb.dtype != np.uint8
        or fused_rgb.ndim != 3
        or fused_rgb.shape[2] != 3
    ):
        raise ValueError("fused image must be a height x width x 3 uint8 RGB array")
    if fused_rgb.shape[:2] != (plan.source_height, plan.source_width):
        raise ValueError("fused image dimensions do not match the packing source")
    if not 0 <= gutter_fill <= 255:
        raise ValueError("gutter_fill must be in [0, 255]")
    canvas = np.full(
        (plan.canvas_height, plan.canvas_width, 3), gutter_fill, dtype=np.uint8
    )
    for placement in plan.placements:
        source, content = placement.source_box, placement.content_box
        crop = fused_rgb[source.y0 : source.y1, source.x0 : source.x1]
        interpolation = (
            cv2.INTER_AREA
            if content.width < crop.shape[1] or content.height < crop.shape[0]
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            crop, (content.width, content.height), interpolation=interpolation
        )
        canvas[content.y0 : content.y1, content.x0 : content.x1] = resized
    return canvas


def _ring_contrast(gray: np.ndarray, source: Rect) -> float:
    center_x = (source.x0 + source.x1) / 2
    center_y = (source.y0 + source.y1) / 2
    outer = Rect(
        max(0, int(np.floor(center_x - 0.75 * source.width))),
        max(0, int(np.floor(center_y - 0.75 * source.height))),
        min(gray.shape[1], int(np.ceil(center_x + 0.75 * source.width))),
        min(gray.shape[0], int(np.ceil(center_y + 0.75 * source.height))),
    )
    values = gray[outer.y0 : outer.y1, outer.x0 : outer.x1]
    ring_mask = np.ones(values.shape, dtype=bool)
    ring_mask[
        source.y0 - outer.y0 : source.y1 - outer.y0,
        source.x0 - outer.x0 : source.x1 - outer.x0,
    ] = False
    ring = values[ring_mask]
    if not ring.size:
        return 0.0
    median = float(np.median(ring))
    mad = float(np.median(np.abs(ring.astype(np.float64) - median)))
    crop_median = float(np.median(gray[source.y0 : source.y1, source.x0 : source.x1]))
    contrast = abs(crop_median - median) / max(1.4826 * mad, 1.0)
    return contrast / (1.0 + contrast)


def _crop_quality(
    gray_image: np.ndarray, box: BoxIndex, placement: TilePlacement
) -> CropQuality:
    source = placement.source_box
    gray_crop = gray_image[source.y0 : source.y1, source.x0 : source.x1]
    fused_contrast = _ring_contrast(gray_image, source)
    laplacian_variance = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    sharpness = laplacian_variance / (laplacian_variance + 400.0)
    histogram = np.bincount(gray_crop.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / max(1, gray_crop.size)
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 8.0)
    scale_retention = min(1.0, float(np.sqrt(placement.scale_x * placement.scale_y)))
    a_score = float(box.quality.score)
    base_score = (
        0.40 * a_score + 0.25 * fused_contrast + 0.20 * entropy + 0.15 * sharpness
    )
    score = base_score * (0.75 + 0.25 * scale_retention)
    return CropQuality(
        score=float(np.clip(score, 0.0, 1.0)),
        a_score=a_score,
        ir_contrast=float(box.quality.ir_contrast),
        fused_contrast=fused_contrast,
        sharpness=sharpness,
        entropy=entropy,
        scale_retention=scale_retention,
        proposal_count=box.quality.proposal_count,
    )


def score_crop_placements(
    fused_rgb: np.ndarray, boxes: Sequence[BoxIndex], plan: PackingPlan
) -> PackingPlan:
    """Attach reproducible A/B quality metadata to every placed crop."""

    by_index = {box.index: box for box in boxes}
    gray_image = cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2GRAY)
    placements = tuple(
        replace(
            placement,
            quality=_crop_quality(gray_image, by_index[placement.index], placement),
        )
        for placement in plan.placements
    )
    return replace(plan, placements=placements)


def canvas_to_source_edge(
    u: float, v: float, tile: TilePlacement
) -> tuple[float, float]:
    return tile.source_box.x0 + (
        u - tile.content_box.x0
    ) / tile.scale_x, tile.source_box.y0 + (v - tile.content_box.y0) / tile.scale_y


def source_to_canvas_edge(
    x: float, y: float, tile: TilePlacement
) -> tuple[float, float]:
    return tile.content_box.x0 + (
        x - tile.source_box.x0
    ) * tile.scale_x, tile.content_box.y0 + (y - tile.source_box.y0) * tile.scale_y


def reproject_tile_mask(
    tile_mask: np.ndarray, tile: TilePlacement, source_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject one content-box mask and return its observed-source validity."""

    if tile_mask.ndim != 2 or tile_mask.shape != (
        tile.content_box.height,
        tile.content_box.width,
    ):
        raise ValueError("tile mask must match the placement content rectangle")
    source_height, source_width = source_shape
    source = tile.source_box
    if not (
        0 <= source.x0 < source.x1 <= source_width
        and 0 <= source.y0 < source.y1 <= source_height
    ):
        raise ValueError("tile source rectangle is out of bounds")
    was_bool = tile_mask.dtype == np.bool_
    resized = cv2.resize(
        tile_mask.astype(np.uint8) if was_bool else tile_mask,
        (source.width, source.height),
        interpolation=cv2.INTER_NEAREST,
    )
    if was_bool:
        resized = resized.astype(bool)
    output = np.zeros((source_height, source_width), dtype=resized.dtype)
    validity = np.zeros((source_height, source_width), dtype=bool)
    output[source.y0 : source.y1, source.x0 : source.x1] = resized
    validity[source.y0 : source.y1, source.x0 : source.x1] = True
    return output, validity
