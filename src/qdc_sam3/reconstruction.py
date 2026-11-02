"""Algorithm D: quality-weighted mask reconstruction in source coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .packing import TilePlacement, reproject_tile_mask


@dataclass(frozen=True)
class QWCRConfig:
    """Controls SAM3 evidence extraction and quality-weighted reconstruction."""

    sam3_output: str = "confidence"
    local_strength: float = 2.0
    quality_min: float = 0.25
    pixel_threshold: float = 0.5
    output_threshold: float = 0.5


@dataclass(frozen=True)
class TileEvidence:
    index: int
    id: str
    quality: float
    used: bool
    foreground_pixels: int
    mean_confidence: float
    contributed_pixels: int

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "index": self.index,
            "id": self.id,
            "quality": self.quality,
            "used": self.used,
            "foreground_pixels": self.foreground_pixels,
            "mean_confidence": self.mean_confidence,
            "contributed_pixels": self.contributed_pixels,
        }


@dataclass(frozen=True)
class ReconstructionResult:
    mask: np.ndarray
    crop_mask: np.ndarray
    fused_confidence: np.ndarray
    crop_confidence: np.ndarray
    crop_reliability: np.ndarray
    final_confidence: np.ndarray
    tile_evidence: tuple[TileEvidence, ...]


def validate_qwcr_config(config: QWCRConfig) -> None:
    if config.sam3_output not in {"binary", "confidence"}:
        raise ValueError("sam3_output must be binary or confidence")
    if config.local_strength < 0:
        raise ValueError("local_strength must be non-negative")
    for name, value in (
        ("quality_min", config.quality_min),
        ("pixel_threshold", config.pixel_threshold),
        ("output_threshold", config.output_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _reproject_confidence(
    tile_confidence: np.ndarray,
    placement: TilePlacement,
    source_shape: tuple[int, int],
    *,
    binary: bool,
) -> tuple[np.ndarray, np.ndarray]:
    source = placement.source_box
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    resized = cv2.resize(
        tile_confidence.astype(np.float32),
        (source.width, source.height),
        interpolation=interpolation,
    )
    projected = np.zeros(source_shape, dtype=np.float32)
    validity = np.zeros(source_shape, dtype=bool)
    projected[source.y0 : source.y1, source.x0 : source.x1] = resized
    validity[source.y0 : source.y1, source.x0 : source.x1] = True
    return projected, validity


def quality_weighted_reprojection(
    fused_confidence: np.ndarray,
    packed_confidence: np.ndarray,
    packed_mask: np.ndarray,
    placements: tuple[TilePlacement, ...],
    source_shape: tuple[int, int],
    config: QWCRConfig,
) -> ReconstructionResult:
    """Fuse global and per-tile SAM3 evidence using bounded A/B quality."""

    validate_qwcr_config(config)
    if fused_confidence.shape != source_shape:
        raise ValueError("fused confidence must use source coordinates")
    if packed_confidence.shape != packed_mask.shape:
        raise ValueError("packed confidence and mask must align")

    numerator = np.zeros(source_shape, dtype=np.float32)
    denominator = np.zeros(source_shape, dtype=np.float32)
    reliability_complement = np.ones(source_shape, dtype=np.float32)
    crop_mask = np.zeros(source_shape, dtype=bool)
    evidence: list[TileEvidence] = []
    use_binary = config.sam3_output == "binary"

    for placement in placements:
        content = placement.content_box
        tile_confidence = packed_confidence[
            content.y0 : content.y1, content.x0 : content.x1
        ]
        tile_mask = packed_mask[content.y0 : content.y1, content.x0 : content.x1]
        projected_confidence, validity = _reproject_confidence(
            tile_confidence, placement, source_shape, binary=use_binary
        )
        projected_mask, _ = reproject_tile_mask(tile_mask, placement, source_shape)
        crop_mask |= projected_mask.astype(bool)

        quality = 1.0 if placement.quality is None else placement.quality.score
        quality = float(np.clip(quality, 0.0, 1.0))
        used = quality >= config.quality_min
        if used:
            numerator[validity] += quality * projected_confidence[validity]
            denominator[validity] += quality
            reliability_complement[validity] *= 1.0 - quality
        evidence.append(
            TileEvidence(
                index=placement.index,
                id=placement.id,
                quality=quality,
                used=used,
                foreground_pixels=int(projected_mask.sum()),
                mean_confidence=(
                    float(projected_confidence[validity].mean())
                    if validity.any()
                    else 0.0
                ),
                contributed_pixels=(
                    int(
                        np.count_nonzero(
                            projected_confidence[validity] >= config.pixel_threshold
                        )
                    )
                    if used
                    else 0
                ),
            )
        )

    crop_confidence = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    crop_reliability = np.where(
        denominator > 0, 1.0 - reliability_complement, 0.0
    ).astype(np.float32)
    local_weight = config.local_strength * crop_reliability
    final_confidence = (
        np.clip(fused_confidence, 0.0, 1.0) + local_weight * crop_confidence
    ) / (1.0 + local_weight)
    mask = final_confidence >= config.output_threshold
    return ReconstructionResult(
        mask=mask,
        crop_mask=crop_mask,
        fused_confidence=fused_confidence.astype(np.float32, copy=False),
        crop_confidence=crop_confidence,
        crop_reliability=crop_reliability,
        final_confidence=final_confidence.astype(np.float32, copy=False),
        tile_evidence=tuple(evidence),
    )
