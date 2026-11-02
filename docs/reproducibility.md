# Reproducing the full model

## Checkpoint and backend identity

No checkpoints or fusion models are distributed. Fusion is supplied externally
as a callable mapping registered IR/VI arrays to fused RGB pixels. Record the
identity, configuration and weights of your chosen external backend separately.

| Component | Identity |
| --- | --- |
| SAM3 code revision | `99d02f28c8d34610ab35559b72642b01f10fdb1e` |
| SAM3 checkpoint SHA-256 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| Validated runtime | Python 3.12.3, PyTorch 2.5.1+cu121 |

New upstream commits/checkpoints can change results even if the QDC-SAM3
configuration is unchanged. Record their identities rather than silently
substituting another model.

## Episode protocol

Input VI and IR pairs must be spatially registered. Support/query identities
must be distinct, and support masks must be class-specific foreground masks.
The supplied episode generator retains the benchmark class IDs and text
mapping.

Full 1-shot: select the largest foreground mask per class (descending filename
for area ties); evaluate all remaining class-labeled examples.

Full 5-shot: choose the five largest masks per class (ascending filename ties),
place all five on each shared support canvas, and exclude all five from queries.
Paired 1-shot uses the first of those five supports while keeping the same
five-support exclusion/query set.

| Dataset | Full 1-shot queries | Paired 1-shot / 5-shot queries |
| --- | ---: | ---: |
| Tokyo-4i | 1,106 | 1,042 |
| VT-840-5i | 820 | 740 |

The metric aggregates foreground intersection and union across queries
**within each class**, divides these totals, then averages class IoUs.
It is neither foreground/background IoU nor arithmetic mean of per-image IoUs.
The prompt text is part of the protocol and must be retained in comparisons.

## Release verification

The release is checked with 43 weight-free unit tests, standalone wheel
installation, CLI help and distribution-content audits.

Actual CUDA inference was compared against the frozen full implementation
for one Tokyo human query and one VT bottle query, both at 1-shot and 5-shot.
The fusion model was injected externally for this equivalence check; it is
not included in this release. An arbitrary replacement fusion backend is not
guaranteed to reproduce the paper's numerical metrics.
The final masks, final/branch confidences, fused pixels, packed IR pixels,
shared canvases, proposal provenance and placement/quality metadata matched
exactly. This is an implementation-equivalence smoke test, not a new full
benchmark evaluation.

The release contains no ablation-only processing, legacy single-forward interfaces,
machine-specific experiment schedulers, private datasets or bundled weights.
