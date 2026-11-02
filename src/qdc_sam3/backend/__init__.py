"""Pluggable fusion and segmentation backends."""

from .fusion import CallableFusion, load_fusion_backend
from .interfaces import FusionBackend, SegmentationBackend, SegmentationEvidence
from .sam3 import SAM3Backend

__all__ = [
    "FusionBackend",
    "CallableFusion",
    "load_fusion_backend",
    "SAM3Backend",
    "SegmentationBackend",
    "SegmentationEvidence",
]
