"""Backend contracts; no model dependencies are imported here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..packing import Rect
from ..reconstruction import QWCRConfig


@dataclass(frozen=True)
class SegmentationEvidence:
    """Canvas-coordinate, instance-aggregated evidence."""

    confidence: np.ndarray
    mask: np.ndarray
    scores: tuple[float, ...] = ()
    initial_threshold: float | None = None
    effective_threshold: float | None = None
    fallback_applied: bool = False


class FusionBackend(Protocol):
    def __call__(self, visible: np.ndarray, infrared: np.ndarray) -> np.ndarray:
        """Return aligned H x W x 3 RGB uint8 pixels."""
        ...


class SegmentationBackend(Protocol):
    def predict(
        self,
        canvas: np.ndarray,
        support_boxes: tuple[Rect, ...],
        text_prompt: str | None,
        query_panels: tuple[Rect, ...],
        evidence_config: QWCRConfig,
    ) -> SegmentationEvidence:
        """Return float32 confidence and bool mask, both canvas H x W."""
        ...
