"""QDC-SAM3: quality-guided dual-canvas infrared-visible few-shot segmentation."""

from .pipeline import QDCSAM3, Prediction, QDCConfig, Support
from .proposals import BoxIndex, BoxQuality, DISGConfig, predict_boxes
from .reconstruction import QWCRConfig

__all__ = [
    "QDCSAM3",
    "BoxIndex",
    "BoxQuality",
    "DISGConfig",
    "Prediction",
    "QDCConfig",
    "QWCRConfig",
    "Support",
    "predict_boxes",
]
__version__ = "1.0.0"
