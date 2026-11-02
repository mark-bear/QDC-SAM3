"""Evaluate explicit episodes; aggregate foreground intersection/union by class."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import QDCSAM3
from .backend import SAM3Backend, load_fusion_backend
from .cli import load_episode


def summarize(records):
    counts = defaultdict(lambda: [0, 0])
    for record in records:
        values = counts[str(record["class_id"])]
        values[0] += record["intersection"]
        values[1] += record["union"]
    if not counts:
        raise ValueError("evaluation must contain at least one episode")
    ious = {
        key: intersection / union if union else 1.0
        for key, (intersection, union) in sorted(counts.items())
    }
    return {
        "episodes": len(records),
        "class_iou": ious,
        "miou": float(np.mean(list(ious.values()))),
        "metric": "class-averaged-foreground-iou",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", required=True, help="JSON list of episode manifest paths"
    )
    parser.add_argument(
        "--fusion", required=True, help="User fusion plugin: module:callable"
    )
    parser.add_argument("--sam3-checkpoint", required=True)
    parser.add_argument("--sam3-bpe", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    index = Path(args.episodes)
    manifests = json.loads(index.read_text(encoding="utf-8"))
    fusion = load_fusion_backend(args.fusion)
    model = QDCSAM3(
        SAM3Backend.from_checkpoint(
            checkpoint_path=args.sam3_checkpoint,
            bpe_path=args.sam3_bpe,
            device=args.device,
        ),
        fusion=fusion,
    )
    records = []
    for entry in manifests:
        path = index.parent / entry
        data = json.loads(path.read_text(encoding="utf-8"))
        visible, infrared, supports = load_episode(path)
        prediction = model.predict_multi_support_canvas(
            visible, infrared, supports
        ).mask
        with Image.open(path.parent / data["query"]["mask"]) as image:
            target = np.array(image.convert("L")) > 0
        if prediction.shape != target.shape:
            raise ValueError(f"GT shape mismatch: {path}")
        records.append(
            {
                "episode": entry,
                "class_id": data["class_id"],
                "intersection": int(np.logical_and(prediction, target).sum()),
                "union": int(np.logical_or(prediction, target).sum()),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summarize(records), "records": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
