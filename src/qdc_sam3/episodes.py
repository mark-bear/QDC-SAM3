"""Generate deterministic paper-protocol episodes without copying datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

TOKYO_PROMPTS = {
    1: "car",
    2: "car",
    3: "human",
    4: "human",
    5: "curb",
    6: "curb",
    7: "speed bump",
    8: "speed bump",
    9: "car stop",
    10: "car stop",
    11: "traffic cone",
    12: "traffic cone",
    13: "bicycle",
    14: "bicycle",
    15: "guardrail",
    16: "guardrail",
}
VT840_PROMPTS = {
    1: "pen",
    2: "pillow",
    3: "spatula",
    4: "bottle",
    5: "book",
    6: "eraser",
    7: "USB flash drive",
    8: "potted plant",
    9: "plate",
    10: "cup",
    11: "ladle",
    12: "umbrella",
    13: "trash can",
    14: "thermos",
    15: "chair",
    16: "slippers",
    17: "detergent bottle",
    18: "paper",
    19: "tissue box",
    20: "bowl",
}


def make_episodes(dataset, root, output, shots, paired=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    prompts = TOKYO_PROMPTS if dataset == "tokyo" else VT840_PROMPTS
    suffix = ".png" if dataset == "tokyo" else ".jpg"
    output.mkdir(parents=True, exist_ok=True)
    index = []

    def pair(mask):
        images = root / "seperated_images"
        return {
            "visible": os.path.relpath(images / (mask.stem + "_rgb" + suffix), output),
            "infrared": os.path.relpath(images / (mask.stem + "_th" + suffix), output),
            "mask": os.path.relpath(mask, output),
        }

    for class_id, prompt in prompts.items():
        masks = sorted((root / "Binary_map" / str(class_id)).glob("*.png"))
        if len(masks) <= (5 if paired else shots):
            raise ValueError(f"class {class_id} has insufficient samples")
        areas = []
        for mask in masks:
            with Image.open(mask) as image:
                area = int((np.array(image.convert("L")) > 0).sum())
            areas.append((area, mask))
        if shots == 1 and not paired:
            ranked = [
                p
                for area, p in sorted(
                    areas, key=lambda x: (x[0], x[1].name), reverse=True
                )
            ]
        else:
            ranked = [p for area, p in sorted(areas, key=lambda x: (-x[0], x[1].name))]
        supports = ranked[:shots]
        excluded = set(ranked[:5] if paired else supports)
        for query in masks:
            if query in excluded:
                continue
            name = f"class-{class_id:02d}-{query.stem}.json"
            data = {
                "class_id": f"{dataset}:{class_id}",
                "text_prompt": prompt,
                "query": pair(query),
                "supports": [pair(p) for p in supports],
            }
            (output / name).write_text(json.dumps(data, indent=2) + "\n")
            index.append(name)
    (output / "episodes.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tokyo", "vt"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shots", type=int, choices=[1, 5], required=True)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="exclude all top-five supports for paired 1/5-shot comparisons",
    )
    args = parser.parse_args(argv)
    make_episodes(args.dataset, args.root, args.output, args.shots, args.paired)


if __name__ == "__main__":
    main()
