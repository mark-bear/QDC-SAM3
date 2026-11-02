from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from qdc_sam3 import DISGConfig, predict_boxes
from qdc_sam3.proposals import (
    _Box,
    _g2,
    _gate_branch,
    _postprocess,
    _Processed,
    _Proposal,
)


def _item(identifier: str, box: _Box) -> _Processed:
    return _Processed(_Proposal(identifier, box), box, 0.5)


def test_g2_groups_nearby_fragments_and_preserves_sources() -> None:
    image = np.full((64, 64), 100, dtype=np.uint8)
    result = _g2(
        image,
        [_item("p1-a", _Box(10, 10, 20, 20)), _item("p3-b", _Box(23, 10, 33, 20))],
        replace(DISGConfig(), lambda_distance=0.5),
    )
    assert len(result) == 1
    assert result[0].box == _Box(10, 10, 33, 20)
    assert result[0].proposal.source_ids == ("p1-a", "p3-b")


def test_g2_iterative_merge_closes_online_ordering_gap() -> None:
    image = np.full((64, 64), 100, dtype=np.uint8)
    items = [
        _item("a", _Box(0, 0, 10, 10)),
        _item("b", _Box(30, 0, 40, 10)),
        _item("c", _Box(8, 0, 20, 10)),
    ]
    config = replace(DISGConfig(), lambda_distance=1.2)
    assert len(_g2(image, items, replace(config, iterative_merge=False))) == 2
    result = _g2(image, items, config)
    assert len(result) == 1
    assert result[0].proposal.source_ids == ("a", "b", "c")


def test_expansion_deduplication_preserves_all_provenance() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    result = _postprocess(
        image,
        [_Proposal("small", _Box(0, 0, 9, 9)), _Proposal("full", _Box(0, 0, 10, 10))],
        DISGConfig(),
    )
    assert len(result) == 1
    assert result[0].box == _Box(0, 0, 10, 10)
    assert result[0].proposal.source_ids == ("full", "small")


def test_branch_gate_is_independent_and_quality_ranked() -> None:
    image = np.full((32, 32), 20, dtype=np.uint8)
    image[2:8, 2:8] = 220
    image[20:26, 20:26] = 120
    proposals = [
        _Proposal("weak", _Box(20, 20, 26, 26)),
        _Proposal("strong", _Box(2, 2, 8, 8)),
    ]
    assert [item.id for item in _gate_branch(image, proposals, 1)] == ["strong"]


@pytest.mark.parametrize(
    "image, config, error",
    [
        (np.zeros((8, 8), dtype=np.float32), DISGConfig(), ValueError),
        (np.zeros((0, 8), dtype=np.uint8), DISGConfig(), ValueError),
        (
            np.zeros((8, 8), dtype=np.uint8),
            replace(DISGConfig(), branch_proposal_cap=0),
            ValueError,
        ),
        (np.zeros((8, 8), dtype=np.uint8), object(), TypeError),
    ],
)
def test_public_api_rejects_invalid_input(
    image: np.ndarray, config: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        predict_boxes(image, config)  # type: ignore[arg-type]


def test_python_api_works_when_caller_is_read_from_stdin() -> None:
    code = """
import numpy as np
from qdc_sam3 import predict_boxes
boxes = predict_boxes(np.full((64, 64), 127, dtype=np.uint8))
assert all(0 <= b.x0 < b.x1 <= 64 and 0 <= b.y0 < b.y1 <= 64 for b in boxes)
print(len(boxes))
"""
    completed = subprocess.run(
        [sys.executable, "-"],
        input=code,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert int(completed.stdout.strip()) >= 0


def test_public_boxes_retain_quality_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.full((32, 32), 20, dtype=np.uint8)
    image[8:16, 8:16] = 220
    monkeypatch.setattr(
        "qdc_sam3.proposals._p1",
        lambda ir, config: [_Proposal("p1-a", _Box(8, 8, 16, 16))],
    )
    monkeypatch.setattr("qdc_sam3.proposals._bounded_p3", lambda ir, config: [])
    boxes = predict_boxes(image, replace(DISGConfig(), context_factor=1.0))
    assert len(boxes) == 1
    assert 0.0 <= boxes[0].quality.score <= 1.0
    assert boxes[0].quality.ir_contrast > 0.0
    assert boxes[0].quality.proposal_count == 1
    assert boxes[0].as_dict()["quality"]["algorithm"] == "ir-ring-contrast-v1"
