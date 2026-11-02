import pytest

from qdc_sam3.evaluate import summarize


def test_aggregate_class_iou_not_mean_image_iou():
    result = summarize(
        [
            {"class_id": 1, "intersection": 1, "union": 1},
            {"class_id": 1, "intersection": 0, "union": 9},
            {"class_id": 2, "intersection": 1, "union": 2},
        ]
    )
    assert result["class_iou"] == {"1": 0.1, "2": 0.5}
    assert result["miou"] == pytest.approx(0.3)


def test_empty_evaluation_is_an_error():
    with pytest.raises(ValueError):
        summarize([])
