import numpy as np
import pytest

from qdc_sam3 import BoxIndex, BoxQuality
from qdc_sam3.packing import (
    CanvasConfig,
    canvas_to_source_edge,
    pack_boxes,
    render_canvas,
    reproject_tile_mask,
    score_crop_placements,
    source_to_canvas_edge,
)


def box(index=0):
    return BoxIndex(index, f"box-{index}", 2, 3, 6, 7, ("p1",), BoxQuality(0.8, 0.7, 2))


def test_crop_pixels_and_quality_are_separate():
    boxes = [box()]
    plan = pack_boxes(
        boxes, (10, 10), CanvasConfig(width=8, height=8, gutter=1, scales=(1.0,))
    )
    ir_rgb = np.full((10, 10, 3), 123, dtype=np.uint8)
    fused = np.arange(300, dtype=np.uint8).reshape(10, 10, 3)
    plan = score_crop_placements(fused, boxes, plan)
    image = render_canvas(ir_rgb, plan)
    tile = plan.placements[0]
    c = tile.content_box
    assert np.all(image[c.y0 : c.y1, c.x0 : c.x1] == 123)
    assert tile.quality.a_score == pytest.approx(0.8)
    assert tile.quality.proposal_count == 2


def test_geometry_round_trip_and_mask_reprojection():
    plan = pack_boxes(
        [box()], (10, 10), CanvasConfig(width=12, height=12, gutter=1, scales=(2.0,))
    )
    tile = plan.placements[0]
    u, v = source_to_canvas_edge(3.25, 4.5, tile)
    assert canvas_to_source_edge(u, v, tile) == pytest.approx((3.25, 4.5))
    mask, validity = reproject_tile_mask(
        np.ones((tile.content_box.height, tile.content_box.width), dtype=bool),
        tile,
        (10, 10),
    )
    assert mask.dtype == bool
    assert mask.sum() == validity.sum() == 16


def test_drops_lowest_ranked_index_when_no_scale_fits():
    plan = pack_boxes(
        [box(0), box(1)],
        (10, 10),
        CanvasConfig(width=4, height=4, gutter=0, scales=(1.0,)),
    )
    assert [p.index for p in plan.placements] == [0]
    assert plan.dropped_indices == (1,)


def test_empty_plan_uses_input_dimensions():
    plan = pack_boxes([], (9, 13))
    assert (plan.canvas_height, plan.canvas_width) == (9, 13)
    assert not plan.placements


def test_unordered_box_indices_are_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        pack_boxes([box(1)], (10, 10))
