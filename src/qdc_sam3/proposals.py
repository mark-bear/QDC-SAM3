"""DISG: infrared foreground discovery and compact proposal grouping."""

from __future__ import annotations

import hashlib
import math
import pickle
import subprocess
import sys
from dataclasses import dataclass, replace
from itertools import pairwise

import cv2
import numpy as np


@dataclass(frozen=True, order=True)
class _Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: _Box) -> float:
        width = max(0, min(self.x1, other.x1) - max(self.x0, other.x0))
        height = max(0, min(self.y1, other.y1) - max(self.y0, other.y0))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True)
class _Proposal:
    id: str
    box: _Box
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Processed:
    proposal: _Proposal
    box: _Box
    score: float


@dataclass(frozen=True)
class DISGConfig:
    """Frozen Algorithm A proposal and grouping settings."""

    lambda_distance: float = 1.20
    context_factor: float = 1.25
    branch_proposal_cap: int = 32
    iterative_merge: bool = True
    p1_delta: int = 5
    p1_min_area: int = 16
    p1_max_area_fraction: float = 0.25
    p1_max_variation: float = 0.25
    p3_min_area: int = 16
    p3_max_area_fraction: float = 0.25
    p3_min_side: int = 3
    p3_timeout_seconds: float = 60.0
    min_gap_pixels: int = 12
    gap_fraction: float = 0.50
    max_enclosure_over_union: float = 2.0
    max_fraction: float = 0.35
    cluster_bonus: float = 0.04


_DEFAULT_CONFIG = DISGConfig()


@dataclass(frozen=True)
class BoxQuality:
    """Algorithm A quality evidence for one grouped IR proposal."""

    score: float
    ir_contrast: float
    proposal_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "algorithm": "ir-ring-contrast-v1",
            "score": self.score,
            "ir_contrast": self.ir_contrast,
            "proposal_count": self.proposal_count,
        }


@dataclass(frozen=True)
class BoxIndex:
    """Stable result in half-open source coordinates [x0,x1) x [y0,y1)."""

    index: int
    id: str
    x0: int
    y0: int
    x1: int
    y1: int
    source_ids: tuple[str, ...]
    quality: BoxQuality = BoxQuality(0.0, 0.0, 0)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "id": self.id,
            "box": {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1},
            "source_ids": list(self.source_ids),
            "quality": self.quality.as_dict(),
        }


def _validate_config(config: DISGConfig) -> None:
    if not isinstance(config, DISGConfig):
        raise TypeError("config must be an DISGConfig")
    if (
        config.lambda_distance <= 0
        or config.context_factor <= 0
        or config.branch_proposal_cap < 1
    ):
        raise ValueError(
            "lambda, context factor, and branch proposal cap must be positive"
        )
    if config.p1_delta < 1 or config.p1_min_area < 1:
        raise ValueError("P1 delta and minimum area must be positive")
    if not 0 < config.p1_max_area_fraction <= 1 or config.p1_max_variation < 0:
        raise ValueError("invalid P1 area fraction or variation")
    if config.p3_min_area < 1 or config.p3_min_side < 1:
        raise ValueError("P3 minimum area and side must be positive")
    if not 0 < config.p3_max_area_fraction <= 1 or config.p3_timeout_seconds <= 0:
        raise ValueError("invalid P3 area fraction or timeout")
    if config.min_gap_pixels < 0 or config.gap_fraction < 0:
        raise ValueError("G2 gap limits must be non-negative")
    if config.max_enclosure_over_union < 1 or not 0 < config.max_fraction <= 1:
        raise ValueError("invalid G2 enclosure limits")
    if config.cluster_bonus < 0:
        raise ValueError("cluster bonus must be non-negative")
    if not isinstance(config.iterative_merge, bool):
        raise TypeError("iterative_merge must be a boolean")


def _score(ir: np.ndarray, box: _Box) -> float:
    height, width = ir.shape
    center_x, center_y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    outer = _Box(
        max(0, math.floor(center_x - 0.75 * box.width)),
        max(0, math.floor(center_y - 0.75 * box.height)),
        min(width, math.ceil(center_x + 0.75 * box.width)),
        min(height, math.ceil(center_y + 0.75 * box.height)),
    )
    values = ir[outer.y0 : outer.y1, outer.x0 : outer.x1]
    mask = np.ones(values.shape, dtype=bool)
    mask[
        box.y0 - outer.y0 : box.y1 - outer.y0, box.x0 - outer.x0 : box.x1 - outer.x0
    ] = False
    ring = values[mask]
    if not ring.size:
        return 0.0
    median = float(np.median(ring))
    mad = float(np.median(np.abs(ring.astype(np.float64) - median)))
    box_median = float(np.median(ir[box.y0 : box.y1, box.x0 : box.x1]))
    contrast = abs(box_median - median) / max(1.4826 * mad, 1.0)
    return contrast / (1.0 + contrast)


def _expand(box: _Box, factor: float, width: int, height: int) -> _Box:
    center_x, center_y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    half_width, half_height = box.width * factor / 2, box.height * factor / 2
    return _Box(
        max(0, math.floor(center_x - half_width)),
        max(0, math.floor(center_y - half_height)),
        min(width, math.ceil(center_x + half_width)),
        min(height, math.ceil(center_y + half_height)),
    )


def _p1(ir: np.ndarray, config: DISGConfig) -> list[_Proposal]:
    height, width = ir.shape
    minimum = max(1, round(config.p1_min_area * height * width / (640 * 480)))
    maximum = int(config.p1_max_area_fraction * height * width)
    detector = cv2.MSER_create(
        config.p1_delta, minimum, maximum, config.p1_max_variation
    )
    regions, boxes = detector.detectRegions(ir)
    result = []
    for index, (region, raw) in enumerate(zip(regions, boxes, strict=True)):
        x, y, box_width, box_height = map(int, raw)
        if minimum <= len(region) <= maximum and box_width >= 3 and box_height >= 3:
            result.append(
                _Proposal(f"p1-r{index}", _Box(x, y, x + box_width, y + box_height))
            )
    return result


def _p3(ir: np.ndarray, config: DISGConfig) -> list[_Proposal]:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError("OpenCV contrib Selective Search is unavailable")
    height, width = ir.shape
    minimum = max(1, round(config.p3_min_area * height * width / (640 * 480)))
    maximum = int(config.p3_max_area_fraction * height * width)
    search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    search.setBaseImage(np.repeat(ir[:, :, None], 3, axis=2))
    search.switchToSelectiveSearchFast()
    result, seen = [], set()
    for index, raw in enumerate(search.process()):
        x, y, box_width, box_height = map(int, raw)
        box = _Box(x, y, x + box_width, y + box_height)
        if (
            box not in seen
            and minimum <= box.area <= maximum
            and box.width >= config.p3_min_side
            and box.height >= config.p3_min_side
        ):
            seen.add(box)
            result.append(_Proposal(f"p3-r{index}", box))
    return result


def _bounded_p3(ir: np.ndarray, config: DISGConfig) -> list[_Proposal]:
    """Run P3 without importing and re-executing the caller's main module."""

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "qdc_sam3._p3_worker"],
            input=pickle.dumps((ir, config), protocol=pickle.HIGHEST_PROTOCOL),
            capture_output=True,
            check=False,
            timeout=config.p3_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"P3 exceeded {config.p3_timeout_seconds}s") from error
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        raise RuntimeError(
            f"P3 worker exited with status {completed.returncode}: {stderr}"
        )
    try:
        ok, value = pickle.loads(completed.stdout)
    except Exception as error:
        raise RuntimeError(
            f"P3 worker returned an invalid response: {stderr}"
        ) from error
    if not ok:
        raise RuntimeError(value)
    return value


def _merge_sources(kept: _Processed, added: _Processed) -> _Processed:
    identifiers = tuple(
        sorted(
            set(
                (kept.proposal.source_ids or (kept.proposal.id,))
                + (added.proposal.source_ids or (added.proposal.id,))
            )
        )
    )
    return replace(kept, proposal=replace(kept.proposal, source_ids=identifiers))


def _gate_branch(
    ir: np.ndarray, proposals: list[_Proposal], cap: int
) -> list[_Proposal]:
    """Retain one generator's highest-quality IR proposals before G2."""

    ranked = sorted(proposals, key=lambda item: (-_score(ir, item.box), item.id))
    return ranked[:cap]


def _postprocess(
    ir: np.ndarray, raw: list[_Proposal], config: DISGConfig
) -> list[_Processed]:
    height, width = ir.shape
    scored = [
        _Processed(proposal, proposal.box, _score(ir, proposal.box)) for proposal in raw
    ]

    def rank(item):
        return -item.score, item.proposal.id

    exact: dict[_Box, _Processed] = {}
    for item in sorted(scored, key=rank):
        exact[item.box] = (
            item if item.box not in exact else _merge_sources(exact[item.box], item)
        )
    kept: list[_Processed] = []
    for item in sorted(exact.values(), key=rank):
        match = next(
            (
                index
                for index, existing in enumerate(kept)
                if item.box.iou(existing.box) >= 0.85
                and min(item.box.area, existing.box.area)
                / max(item.box.area, existing.box.area)
                >= 0.8
            ),
            None,
        )
        if match is None:
            kept.append(item)
        else:
            kept[match] = _merge_sources(kept[match], item)
    unique: dict[_Box, _Processed] = {}
    for item in kept:
        box = _expand(item.box, config.context_factor, width, height)
        expanded = _Processed(item.proposal, box, item.score)
        unique[box] = (
            expanded if box not in unique else _merge_sources(unique[box], expanded)
        )
    return sorted(unique.values(), key=rank)


def _enclose(first: _Box, second: _Box) -> _Box:
    return _Box(
        min(first.x0, second.x0),
        min(first.y0, second.y0),
        max(first.x1, second.x1),
        max(first.y1, second.y1),
    )


def _union_area(boxes: list[_Box]) -> int:
    total = 0
    x_edges = sorted({edge for box in boxes for edge in (box.x0, box.x1)})
    for left, right in pairwise(x_edges):
        end = -1
        for top, bottom in sorted(
            (box.y0, box.y1) for box in boxes if box.x0 < right and box.x1 > left
        ):
            if bottom > end:
                total += (right - left) * (bottom - max(top, end))
                end = bottom
    return total


def _gap(first: _Box, second: _Box) -> float:
    return math.hypot(
        max(0, first.x0 - second.x1, second.x0 - first.x1),
        max(0, first.y0 - second.y1, second.y0 - first.y1),
    )


def _feature(box: _Box, width: int, height: int) -> np.ndarray:
    return np.array(
        (
            (box.x0 + box.x1) / (2 * width),
            (box.y0 + box.y1) / (2 * height),
            math.log(box.width / width),
            math.log(box.height / height),
        ),
        dtype=np.float64,
    )


def _cluster_envelope(members: list[_Processed]) -> _Box:
    envelope = members[0].box
    for member in members[1:]:
        envelope = _enclose(envelope, member.box)
    return envelope


def _cluster_centroid(members: list[_Processed], width: int, height: int) -> np.ndarray:
    return np.mean([_feature(member.box, width, height) for member in members], axis=0)


def _clusters_compatible(
    first: list[_Processed],
    second: list[_Processed],
    width: int,
    height: int,
    config: DISGConfig,
) -> bool:
    first_envelope, second_envelope = (
        _cluster_envelope(first),
        _cluster_envelope(second),
    )
    merged = _enclose(first_envelope, second_envelope)
    maximum_gap = max(
        config.min_gap_pixels,
        config.gap_fraction
        * min(
            max(first_envelope.width, first_envelope.height),
            max(second_envelope.width, second_envelope.height),
        ),
    )
    boxes = [member.box for member in first + second]
    return (
        _gap(first_envelope, second_envelope) <= maximum_gap
        and merged.area <= config.max_fraction * width * height
        and merged.area / _union_area(boxes) <= config.max_enclosure_over_union
    )


def _cluster_distance(
    first: list[_Processed], second: list[_Processed], width: int, height: int
) -> float:
    delta = _cluster_centroid(first, width, height) - _cluster_centroid(
        second, width, height
    )
    return float(np.linalg.norm(delta * np.array((2.0, 2.0, 0.35, 0.35))))


def _converge_g2_clusters(
    clusters: list[list[_Processed]],
    width: int,
    height: int,
    config: DISGConfig,
) -> list[list[_Processed]]:
    """Merge the closest compatible center pair until a fixed point."""

    clusters = [list(members) for members in clusters]
    while True:
        choices = []
        for first_index, first in enumerate(clusters):
            for second_index in range(first_index + 1, len(clusters)):
                second = clusters[second_index]
                if not _clusters_compatible(first, second, width, height, config):
                    continue
                distance = _cluster_distance(first, second, width, height)
                if distance <= config.lambda_distance:
                    choices.append((distance, first_index, second_index))
        if not choices:
            return clusters
        _, first_index, second_index = min(choices)
        clusters[first_index].extend(clusters[second_index])
        del clusters[second_index]


def _g2_clusters(
    ir: np.ndarray, items: list[_Processed], config: DISGConfig
) -> list[list[_Processed]]:
    """Return the memberships formed by gated online DP-Means."""

    height, width = ir.shape
    clusters: list[list[_Processed]] = []
    for item in sorted(
        items, key=lambda candidate: (-candidate.score, candidate.proposal.id)
    ):
        feature, choices = _feature(item.box, width, height), []
        for index, members in enumerate(clusters):
            if not _clusters_compatible(members, [item], width, height, config):
                continue
            centroid = _cluster_centroid(members, width, height)
            distance = float(
                np.linalg.norm((feature - centroid) * np.array((2.0, 2.0, 0.35, 0.35)))
            )
            if distance <= config.lambda_distance:
                choices.append((distance, index))
        if choices:
            clusters[min(choices, key=lambda choice: (choice[0], choice[1]))[1]].append(
                item
            )
        else:
            clusters.append([item])
    return (
        _converge_g2_clusters(clusters, width, height, config)
        if config.iterative_merge
        else clusters
    )


def _g2(
    ir: np.ndarray, items: list[_Processed], config: DISGConfig
) -> list[_Processed]:
    """Quality-seeded, gated online DP-Means grouping."""

    clusters = _g2_clusters(ir, items, config)
    result = []
    for members in clusters:
        if len(members) == 1:
            result.append(members[0])
            continue
        box = members[0].box
        for member in members[1:]:
            box = _enclose(box, member.box)
        source_ids = tuple(
            sorted(
                {
                    source
                    for member in members
                    for source in (member.proposal.source_ids or (member.proposal.id,))
                }
            )
        )
        identifier = (
            "g2-" + hashlib.sha1("\0".join(source_ids).encode()).hexdigest()[:12]
        )
        score = min(
            1.0,
            max(_score(ir, box), max(member.score for member in members))
            + config.cluster_bonus * min(len(members) - 1, 3),
        )
        result.append(_Processed(_Proposal(identifier, box, source_ids), box, score))
    return sorted(result, key=lambda item: (-item.score, item.proposal.id))


def predict_boxes(
    ir_u8: np.ndarray, config: DISGConfig = _DEFAULT_CONFIG
) -> list[BoxIndex]:
    """Infer indexed source boxes from one two-dimensional uint8 IR image."""

    if not isinstance(ir_u8, np.ndarray) or ir_u8.dtype != np.uint8 or ir_u8.ndim != 2:
        raise ValueError("expected a two-dimensional uint8 NumPy IR image")
    if not ir_u8.size or not all(ir_u8.shape):
        raise ValueError("IR image must not be empty")
    _validate_config(config)
    p1 = _gate_branch(ir_u8, _p1(ir_u8, config), config.branch_proposal_cap)
    p3 = _gate_branch(ir_u8, _bounded_p3(ir_u8, config), config.branch_proposal_cap)
    grouped = _g2(ir_u8, _postprocess(ir_u8, p1 + p3, config), config)
    return [
        BoxIndex(
            index=index,
            id=item.proposal.id,
            x0=item.box.x0,
            y0=item.box.y0,
            x1=item.box.x1,
            y1=item.box.y1,
            source_ids=item.proposal.source_ids or (item.proposal.id,),
            quality=BoxQuality(
                score=item.score,
                ir_contrast=_score(ir_u8, item.box),
                proposal_count=len(item.proposal.source_ids or (item.proposal.id,)),
            ),
        )
        for index, item in enumerate(grouped)
    ]
