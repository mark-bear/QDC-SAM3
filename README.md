# QDC-SAM3

Quality-Guided Dual-Canvas SAM3 for Infrared–Visible Few-Shot Segmentation.
Quality-guided dual-canvas few-shot segmentation for aligned infrared-visible
(IR-VI) images. This is the full paper implementation: a global fused semantic
branch and a modality-matched packed infrared branch. Fusion is an external
black box: IR and VI enter, fused pixels leave. No fusion model, weights or
model-specific dependencies are included. SAM3 is frozen; no benchmark-specific
training is performed.

## Installation

Use Python 3.12 for the official SAM3 backend. Install a CUDA-compatible PyTorch
build and the external SAM3 package according to the
[official instructions](https://github.com/facebookresearch/sam3).
Then, from this directory:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The core package itself supports Python 3.10–3.13 and does not import PyTorch or
SAM3 until a model adapter is called. Only one OpenCV distribution should be
installed: this package needs the contrib/headless build for Selective Search.
All weights must be supplied explicitly. See
[backend setup](src/qdc_sam3/backend/README.md).

## Inference

Inputs are registered RGB VI (`uint8[H,W,3]`), grayscale IR
(`uint8[H,W]`), and one or more aligned support pairs with a foreground mask.
Each support may have a different spatial size from the query. The binary
support mask must be nonempty. All supports describe the same class/text prompt.

```python
from qdc_sam3 import QDCSAM3, Support
from qdc_sam3.backend import CallableFusion, SAM3Backend
from my_backend import fuse  # Your own callable: fuse(visible, infrared) -> fused

model = QDCSAM3(
    SAM3Backend.from_checkpoint(
        checkpoint_path="weights/sam3/sam3.pt",
        bpe_path="external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device="cuda",
    ),
    fusion=CallableFusion(fuse),
)
support = Support(support_vi, support_ir, support_mask, "human")
prediction = model(query_vi, query_ir, support)
prediction.save("outputs/example")
# 5-shot: same two image-backbone passes, all five supports on each canvas.
prediction = model.predict_multi_support_canvas(query_vi, query_ir, supports)
```

For file-based inference, edit `examples/episode.json`; paths are relative to
the manifest. Masks must be **class-specific binary images**, not semantic
class-index maps; any nonzero value is interpreted as foreground.

```bash
qdc-sam3 --episode examples/episode.json \
  --fusion my_backend:fuse \
  --sam3-checkpoint weights/sam3/sam3.pt \
  --sam3-bpe external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --device cuda --output outputs/example
```

`--width` and `--height` parameterize the shared canvas; defaults are 1008 each.
The official SAM3 processor still resizes to its internal model resolution, so
increasing the external canvas does not enlarge the backbone input resolution.

## Modules and frozen defaults

| Module | Output | Implementation |
| --- | --- | --- |
| DISG (A) | Compact IR proposals, provenance and quality | `proposals.py` |
| QIRP (B) | Packed IR pixels, placements and crop quality | `packing.py` |
| PIVF (C) | Full aligned fused query via a replaceable backend | `backend/` |
| QWCR (D) | Quality-weighted source-coordinate IR evidence | `reconstruction.py` |

Full module names are Dual-Proposal Infrared Saliency Grouping (DISG),
Quality-Aware Infrared Region Packing (QIRP), Pluggable Infrared–Visible Fusion
(PIVF), and Quality-Weighted Confidence Reprojection (QWCR).

DISG uses MSER (P1) and Selective Search Fast (P3), independently quality-gated
to 32 proposals each. G2 uses weighted center/scale distance with lambda 1.2,
geometric compatibility constraints and repeated compatible-cluster merging to
a fixed point, without a cluster-count budget. This is constrained online
DP-Means plus merging, not unconstrained Lloyd-style DP-Means.

QIRP cuts **IR**, not fused pixels. It tries MaxRects packing at descending
scales, with no rotation, and only drops the lowest-ranked remaining proposals
when no scale fits. Quality is measured on the corresponding **fused** crop,
combining A quality, contrast, entropy, sharpness and scale retention.
All rectangles are top-left, half-open coordinates.

The semantic canvas contains VI supports and the full fused query. The IR
canvas contains IR supports and packed IR query crops. Support regions occupy
60% of canvas height at the bottom; gutter is 2 pixels and fill is 0.
Each support foreground bounding box is a positive SAM3 geometric prompt;
the shared class text is also supplied when available.

SAM3 instance confidence threshold is 0.5. If no query mask is found, the
same cached image state is re-filtered at 0.1 (no extra backbone forward).
Instance-aggregated confidence retains both pixel probabilities and detection
scores. Binary evidence is optional through `QWCRConfig(sam3_output="binary")`.

QWCR rejects crops below quality 0.25, averages overlapping probabilities by
quality and computes reliability by noisy-OR. With strength gamma=2:

```text
IR evidence = gamma * reliability * crop_probability / (1 + gamma * reliability)
final_probability = semantic_probability + (1 - semantic_probability) * IR_evidence
final_mask = final_probability >= 0.5
```

## Outputs

`Prediction` exposes the final binary mask and float32 confidence in original
query coordinates, both branch confidences/masks, fused query, packed IR,
two shared canvases, proposal boxes, packing plan and per-tile quality evidence.
`save()` writes PNGs, lossless float32 NPY maps and `metadata.json`; metadata
retains source IDs, quality components, transformations, dropped boxes and
SAM3 scores/fallback flags. PNG confidence previews are not the quantitative
float32 outputs.

## Evaluation and reproducibility

The paper evaluates 1-shot and single-canvas 5-shot on Tokyo-4i and VT-840-5i.
Supports are chosen deterministically by foreground area; query/support are
distinct. The reported metric is class-averaged foreground IoU (mIoU), not
FB-IoU or mean per-image IoU. The episode evaluator in `evaluate.py` consumes
explicit episode manifests, excludes no examples silently, and aggregates
intersection/union per class before averaging. Use the same episode lists,
prompt mapping, checkpoint files and backend versions for comparisons.

For evaluation, add a dataset-qualified `class_id` and `query.mask` (a binary
GT path) to each episode. Prepare a JSON array of manifest paths, relative to
the index file, and run:

```bash
qdc-sam3-evaluate --episodes episodes.json \
  --fusion my_backend:fuse \
  --sam3-checkpoint weights/sam3/sam3.pt \
  --sam3-bpe external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --output outputs/metrics.json
```

Use separate indexes for separate datasets. The evaluator runs the explicit
episodes supplied by you. Generate the paper's deterministic support/query
protocol from the benchmark's existing class binary maps with:

```bash
qdc-sam3-episodes --dataset tokyo --root /path/to/tokyo/dataset \
  --shots 1 --output episodes/tokyo-1shot
qdc-sam3-episodes --dataset vt --root /path/to/VT-840-5i \
  --shots 5 --output episodes/vt-5shot
```

The dataset layout is `Binary_map/<class_id>/*.png` and
`seperated_images/<stem>_rgb.<ext>`, `<stem>_th.<ext>`, with PNG images for
Tokyo and JPEG images for VT. Existing class IDs/prompts are retained; this tool
does not redefine dataset folds. Single-shot support selection breaks area
ties by descending filename, exactly as the full 1-shot run; five-shot uses
ascending filename ties. Add `--paired` for the paired 1-shot protocol: use
the first of the top-five supports, and exclude all five from queries.
Evaluate the generated `episodes.json` with the command above.

The full 1-shot results below use 1,106 Tokyo and 820 VT queries. Five-shot
excludes five supports per class and uses 1,042 Tokyo and 740 VT queries.
Thus these rows must not be subtracted as a strictly paired shot-count gain.

These are frozen paper-run results, not promised metrics for an arbitrary fusion
backend. Changing the external fused pixels can change predictions.

| Paper full model | Tokyo-4i mIoU (%) | VT-840-5i mIoU (%) |
| --- | ---: | ---: |
| 1-shot | 56.01 | 84.17 |
| 5-shot | 59.61 | 84.92 |

Third-party packages and weights are not redistributed. This directory can be
copied as a standalone repository: it has no private sibling-package imports,
absolute machine paths, datasets, checkpoints, experiment schedulers or
ablation-only switches.

## Attribution and license

Original QDC-SAM3 code is released under the [MIT license](LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md). SAM3 code and model terms
apply independently, as do the terms of any user-supplied fusion model.
