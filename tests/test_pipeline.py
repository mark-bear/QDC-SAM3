from __future__ import annotations

import json

import numpy as np
import pytest

from qdc_sam3 import QDCSAM3, BoxIndex, BoxQuality, QDCConfig, QWCRConfig, Support
from qdc_sam3.backend import SAM3Backend
from qdc_sam3.cli import load_episode


class FakeProcessor:
    device = "cpu"
    confidence_threshold = 0.5

    def __init__(self, empty=False):
        self.images = []
        self.prompt_boxes = []
        self.empty = empty
        self.backbone_calls = 0
        self.threshold_calls = 0

    def set_image(self, image):
        self.image = np.asarray(image)
        self.images.append(self.image.copy())
        self.backbone_calls += 1
        return {}

    def reset_all_prompts(self, state):
        pass

    def set_text_prompt(self, prompt, state):
        state["text"] = prompt
        return state

    def add_geometric_prompt(self, *, state, box, label):
        self.prompt_boxes.append(tuple(box))
        height, width = self.image.shape[:2]
        probability = np.zeros((1, 1, height, width), dtype=np.float32)
        if not self.empty or self.confidence_threshold < 0.5:
            probability[:, :, : round(height * 2 / 3)] = 0.8
        state["masks_logits"] = probability
        state["masks"] = probability >= 0.5
        state["scores"] = np.array([0.9], dtype=np.float32)
        return state

    def set_confidence_threshold(self, threshold, state):
        self.confidence_threshold = threshold
        self.threshold_calls += 1
        probability = np.full((1, 1, *self.image.shape[:2]), 0.8, dtype=np.float32)
        state.update(masks_logits=probability, masks=probability >= 0.5)
        return state


def boxes(ir):
    return [BoxIndex(0, "box-0", 2, 2, 6, 6, ("p1-0",), BoxQuality(0.9, 0.8, 1))]


def inputs():
    vi = np.full((8, 8, 3), 80, dtype=np.uint8)
    ir = np.full((8, 8), 120, dtype=np.uint8)
    svi = np.full((6, 6, 3), 160, dtype=np.uint8)
    sir = np.full((6, 6), 200, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 2:4] = True
    return vi, ir, Support(svi, sir, mask, "human")


def make_pipeline(monkeypatch, processor=None):
    monkeypatch.setattr("qdc_sam3.pipeline.predict_boxes", lambda ir, config: boxes(ir))
    processor = processor or FakeProcessor()
    model = QDCSAM3(
        SAM3Backend(processor),
        fusion=lambda vi, ir: np.full_like(vi, 100),
        config=QDCConfig(width=60, height=60, support_fraction=1 / 3),
        qwcr_config=QWCRConfig(quality_min=0),
    )
    return model, processor


@pytest.mark.parametrize("shots", [1, 5])
def test_two_branches_supports_and_saved_quality(monkeypatch, tmp_path, shots):
    model, processor = make_pipeline(monkeypatch)
    vi, ir, support = inputs()
    result = model.predict_multi_support_canvas(vi, ir, [support] * shots)
    assert processor.backbone_calls == 2
    assert len(processor.prompt_boxes) == 2 * shots
    assert result.mask.shape == ir.shape
    assert result.mask.dtype == bool
    assert result.final_confidence.dtype == np.float32
    assert np.all(result.final_confidence >= result.fused_confidence)
    assert result.packing_plan.canvas_width == 60
    placement = result.packing_plan.placements[0]
    c = placement.content_box
    assert np.all(result.infrared_packed[c.y0 : c.y1, c.x0 : c.x1] == 120)
    metadata = json.loads(result.save(tmp_path)["metadata"].read_text())
    assert metadata["forward_1"]["query"] == ["fused_full"]
    assert metadata["forward_2"]["query"] == ["infrared_packed"]
    assert metadata["support_count"] == shots
    assert metadata["algorithm_b"]["placements"][0]["quality"]["score"] >= 0
    np.testing.assert_array_equal(
        np.load(tmp_path / "final-confidence.npy"), result.final_confidence
    )


def test_empty_boxes_preserve_semantic_branch(monkeypatch):
    model, _ = make_pipeline(monkeypatch)
    monkeypatch.setattr("qdc_sam3.pipeline.predict_boxes", lambda ir, config: [])
    result = model(*inputs())
    assert result.boxes == ()
    np.testing.assert_array_equal(result.final_confidence, result.fused_confidence)


def test_fallback_reuses_backbone_and_restores_threshold(monkeypatch):
    model, processor = make_pipeline(monkeypatch, FakeProcessor(empty=True))
    result = model(*inputs())
    assert result.fused_fallback_applied and result.infrared_fallback_applied
    assert processor.backbone_calls == 2
    assert processor.threshold_calls == 2
    assert processor.confidence_threshold == 0.5


def test_support_validation(monkeypatch):
    model, _ = make_pipeline(monkeypatch)
    vi, ir, support = inputs()
    with pytest.raises(ValueError, match="at least one"):
        model.predict_multi_support_canvas(vi, ir, [])
    with pytest.raises(ValueError, match="foreground"):
        model(
            vi,
            ir,
            Support(
                support.visible, support.infrared, np.zeros_like(support.mask), "human"
            ),
        )


def test_manifest_paths_and_binary_masks(tmp_path):
    from PIL import Image

    vi, ir, support = inputs()
    for name, array in [
        ("vi.png", vi),
        ("ir.png", ir),
        ("svi.png", support.visible),
        ("sir.png", support.infrared),
        ("mask.png", support.mask.astype(np.uint8) * 255),
    ]:
        Image.fromarray(array).save(tmp_path / name)
    data = {
        "query": {"visible": "vi.png", "infrared": "ir.png"},
        "supports": [{"visible": "svi.png", "infrared": "sir.png", "mask": "mask.png"}],
        "text_prompt": "human",
    }
    (tmp_path / "episode.json").write_text(json.dumps(data))
    loaded_vi, loaded_ir, supports = load_episode(tmp_path / "episode.json")
    np.testing.assert_array_equal(loaded_vi, vi)
    np.testing.assert_array_equal(loaded_ir, ir)
    np.testing.assert_array_equal(supports[0].mask, support.mask)
