from __future__ import annotations

import numpy as np

from qdc_sam3 import QWCRConfig
from qdc_sam3.backend.sam3 import extract_sam3_evidence
from qdc_sam3.packing import CropQuality, Rect, TilePlacement
from qdc_sam3.reconstruction import (
    quality_weighted_reprojection,
)


def _placement(quality: float) -> TilePlacement:
    return TilePlacement(
        index=0,
        id="tile-0",
        source_ids=("p1-0",),
        source_box=Rect(0, 0, 2, 2),
        reserved_box=Rect(0, 0, 2, 2),
        content_box=Rect(0, 0, 2, 2),
        scale_x=1.0,
        scale_y=1.0,
        quality=CropQuality(quality, quality, 0.5, 0.5, 0.5, 0.5, 1.0, 1),
    )


def _fuse(fused: float, local: float, quality: float):
    return quality_weighted_reprojection(
        np.full((2, 2), fused, dtype=np.float32),
        np.full((2, 2), local, dtype=np.float32),
        np.full((2, 2), local >= 0.5, dtype=bool),
        (_placement(quality),),
        (2, 2),
        QWCRConfig(),
    )


def test_high_quality_local_evidence_can_add_foreground() -> None:
    result = _fuse(0.0, 1.0, 0.8)
    np.testing.assert_allclose(result.final_confidence, 1.6 / 2.6)
    assert result.mask.all()


def test_low_quality_local_evidence_cannot_override_global() -> None:
    result = _fuse(0.0, 1.0, 0.4)
    assert not result.mask.any()


def test_high_quality_local_background_can_suppress_global() -> None:
    result = _fuse(1.0, 0.0, 0.8)
    np.testing.assert_allclose(result.final_confidence, 1.0 / 2.6)
    assert not result.mask.any()


def test_sam3_probability_and_instance_score_are_preserved() -> None:
    state = {
        "masks_logits": np.full((1, 1, 2, 2), 0.8, dtype=np.float32),
        "masks": np.ones((1, 1, 2, 2), dtype=bool),
        "scores": np.array([0.75], dtype=np.float32),
    }
    confidence, binary = extract_sam3_evidence(
        state,
        (2, 2),
        mode="confidence",
        instance_threshold=0.5,
        pixel_threshold=0.5,
    )
    np.testing.assert_allclose(confidence, 0.6)
    assert binary.all()
