import json

import numpy as np
from PIL import Image

from qdc_sam3.episodes import make_episodes


def test_exact_tie_break_and_query_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr("qdc_sam3.episodes.TOKYO_PROMPTS", {1: "human"})
    root = tmp_path / "dataset"
    directory = root / "Binary_map/1"
    directory.mkdir(parents=True)
    for i in range(6):
        Image.fromarray(np.full((2, 2), 255, dtype=np.uint8)).save(
            directory / f"{i}.png"
        )
    single = tmp_path / "single"
    index = make_episodes("tokyo", root, single, 1)
    assert len(index) == 5
    data = json.loads((single / index[0]).read_text())
    assert data["supports"][0]["mask"].endswith("5.png")
    five = tmp_path / "five"
    index = make_episodes("tokyo", root, five, 5)
    assert len(index) == 1
    data = json.loads((five / index[0]).read_text())
    assert data["query"]["mask"].endswith("5.png")
    assert data["supports"][0]["mask"].endswith("0.png")
