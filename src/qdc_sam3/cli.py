"""Run aligned query/support pairs from an explicit episode manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import QDCSAM3, QDCConfig, Support
from .backend import SAM3Backend, load_fusion_backend


def load_episode(manifest: str | Path):
    """Read paths relative to the JSON manifest, not the working directory."""
    manifest = Path(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    def read(path, mode):
        with Image.open(manifest.parent / path) as image:
            return np.array(image.convert(mode))

    query = data["query"]
    visible, infrared = read(query["visible"], "RGB"), read(query["infrared"], "L")
    supports = tuple(
        Support(
            read(x["visible"], "RGB"),
            read(x["infrared"], "L"),
            read(x["mask"], "L") > 0,
            data.get("text_prompt"),
        )
        for x in data["supports"]
    )
    return visible, infrared, supports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument(
        "--fusion", required=True, help="User fusion plugin: module:callable"
    )
    parser.add_argument("--sam3-checkpoint", required=True)
    parser.add_argument("--sam3-bpe", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--width", type=int, default=1008)
    parser.add_argument("--height", type=int, default=1008)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    visible, infrared, supports = load_episode(args.episode)
    fusion = load_fusion_backend(args.fusion)
    pipeline = QDCSAM3(
        SAM3Backend.from_checkpoint(
            checkpoint_path=args.sam3_checkpoint,
            bpe_path=args.sam3_bpe,
            device=args.device,
        ),
        fusion=fusion,
        config=QDCConfig(width=args.width, height=args.height),
    )
    result = pipeline.predict_multi_support_canvas(visible, infrared, supports)
    result.save(args.output)


if __name__ == "__main__":
    main()
