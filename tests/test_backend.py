import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from qdc_sam3.backend import CallableFusion, SAM3Backend, load_fusion_backend


def test_import_does_not_load_model_frameworks():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,qdc_sam3,qdc_sam3.backend; assert 'torch' not in sys.modules; assert 'sam3' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_explicit_segmentation_weights_are_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        SAM3Backend.from_checkpoint(
            checkpoint_path=tmp_path / "missing.pt",
            bpe_path=tmp_path / "missing.gz",
            device="cpu",
        )


def test_fusion_black_box_preserves_inputs_and_output():
    vi = np.full((8, 8, 3), 80, dtype=np.uint8)
    ir = np.full((8, 8), 120, dtype=np.uint8)
    expected = np.full_like(vi, 99)
    calls = []

    def fuse(visible, infrared):
        calls.append((visible, infrared))
        return expected

    assert CallableFusion(fuse)(vi, ir) is expected
    assert len(calls) == 1 and calls[0][0] is vi and calls[0][1] is ir


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((8, 8, 3), dtype=np.float32),
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        None,
    ],
)
def test_fusion_rejects_invalid_output(bad):
    with pytest.raises(ValueError, match="return aligned"):
        CallableFusion(lambda vi, ir: bad)(
            np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)
        )


def test_fusion_rejects_invalid_inputs_before_calling_backend():
    def forbidden(vi, ir):
        raise AssertionError("backend must not be called")

    fusion = CallableFusion(forbidden)
    with pytest.raises(ValueError, match="visible"):
        fusion(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="infrared"):
        fusion(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))


def test_external_plugin_loader(monkeypatch):
    module = ModuleType("test_external_fusion")
    module.fuse = lambda vi, ir: np.full_like(vi, 100)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = load_fusion_backend("test_external_fusion:fuse")
    assert np.all(
        backend(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))
        == 100
    )


@pytest.mark.parametrize("spec", ["", "module", ":fuse", "module:", "a:b:c"])
def test_plugin_requires_explicit_module_and_callable(spec):
    with pytest.raises(ValueError, match="module:callable"):
        load_fusion_backend(spec)


def test_plugin_rejects_model_classes(monkeypatch):
    module = ModuleType("test_external_fusion")
    module.Model = CallableFusion
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(TypeError, match="not a class"):
        load_fusion_backend("test_external_fusion:Model")
