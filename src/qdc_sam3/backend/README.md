# Fusion and segmentation backends

Both dependencies are external backends. QDC-SAM3 neither trains nor downloads
them automatically, and this release contains no weights or third-party repo.

## Abstract fusion layer

The only contract is **{IR, VI} -> {fused}**. QDC-SAM3 does not select, implement,
initialize or download a fusion model.

For the Python API, implement a callable `fuse(visible, infrared)`:

- `visible`: registered RGB uint8 array of shape H x W x 3.
- `infrared`: registered grayscale uint8 array of shape H x W.
- Return `fused`: aligned RGB uint8 array of shape H x W x 3.

Use `CallableFusion(fuse)` to validate the input/output contract. All device
selection, color conversion, normalization and model/weight loading remain
inside your external implementation. If it produces grayscale fused pixels,
convert them to RGB inside your adapter before returning.

```python
from qdc_sam3.backend import CallableFusion
from my_backend import fuse

fusion = CallableFusion(fuse)
fused = fusion(visible, infrared)
pipeline = QDCSAM3(segmentation, fusion=fusion)
```

The CLI accepts `--fusion my_backend:fuse`. Your module must be installed or
importable on Python's search path. Export a function or an already initialized
callable instance; a class or a no-argument factory is not a plugin. The CLI's
`--device` controls SAM3 only. Configure fusion hardware within your own plugin.

There is no default or identity fallback: an external fusion callable must be
supplied. You may also supply a precomputed fused image through a callable,
provided it corresponds to the input pair and passes the same shape/dtype
contract. The abstract layer does not know the backend's architecture.

## SAM3 access and installation

1. Request access at [facebook/sam3](https://huggingface.co/facebook/sam3)
   and accept its model terms using your own account. Access is gated.
2. Follow [official SAM3 installation](https://github.com/facebookresearch/sam3).
   Install its Python package in the same environment as QDC-SAM3. Keep the
   third-party checkout outside this release, for example `external/sam3/`.
3. Authenticate and download only after access approval:

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download facebook/sam3 sam3.pt config.json --local-dir weights/sam3
```

Never commit your HF token or login cache. The standard image builder uses the
checkpoint plus the tokenizer vocabulary shipped in the official SAM3 package
(`sam3/assets/bpe_simple_vocab_16e6.txt.gz`). Pass both explicitly:

```python
segmentation = SAM3Backend.from_checkpoint(
    checkpoint_path="weights/sam3/sam3.pt",
    bpe_path="external/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    device="cuda",
    confidence_threshold=0.5,
    fallback_confidence_threshold=0.1,
)
```

Do not silently replace the paper SAM3 image checkpoint with SAM3.1 video
weights. Record the exact upstream commit and checkpoint checksum used.
The paper-compatible SAM3 checkout tested with this release is
`99d02f28c8d34610ab35559b72642b01f10fdb1e`.
Pin this revision before installation when reproducing the paper-compatible
backend:

```bash
git clone https://github.com/facebookresearch/sam3.git external/sam3
git -C external/sam3 checkout 99d02f28c8d34610ab35559b72642b01f10fdb1e
python -m pip install -e external/sam3
```

`config.json` is retained for upstream provenance; the explicit Python image
builder used here constructs its architecture in code and does not parse this
JSON. The checkpoint and BPE vocabulary are its required local inputs.

The validated environment uses Python 3.12.3 and PyTorch 2.5.1+cu121 with this
pinned checkout. Current upstream installation recommendations may require
newer PyTorch/CUDA; follow upstream guidance when choosing another revision.

SAM3 runs under CUDA BF16 autocast and inference mode in this adapter.
Its instance probabilities/scores are reduced to canvas-coordinate evidence;
the processor's `masks_logits` field in the tested version contains probabilities
despite its name (do not apply another sigmoid). Threshold fallback operates on
cached backbone state (the tested processor reruns grounding heads, not the
image backbone); processor objects are mutable and must not be shared concurrently
without a lock.

## Another segmentation backend

Implement `SegmentationBackend.predict(canvas, support_boxes, text_prompt,
query_panels, evidence_config)`. Boxes and query panels are half-open integer
rectangles in the **external canvas** coordinates. SAM3-specific normalization
to center-x/center-y/width/height belongs in the SAM3 adapter, not the pipeline.

Return `SegmentationEvidence` with `float32[H,W]` probabilities in [0,1],
`bool[H,W]` masks and optional instance scores/threshold/fallback metadata.
Output H,W must equal the input canvas. If the model resizes internally, restore
these dimensions in your adapter before returning. Confidence/binary mode and
pixel threshold are specified by `QWCRConfig`.

Replacing the segmentation model changes the experimental setting; it is not
an exact reproduction of the SAM3 paper results.
