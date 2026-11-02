"""Model-agnostic IR/VI-to-fused boundary; no fusion model is bundled."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

import numpy as np


class CallableFusion:
    """Wrap a user-owned black box with an explicit array contract.

    The callable receives (visible, infrared) and returns fused RGB pixels.
    Model loading, normalization, weights and device policy belong to the caller.
    """

    def __init__(self, function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        if not callable(function):
            raise TypeError("fusion backend must be callable")
        self.function = function

    def __call__(self, visible: np.ndarray, infrared: np.ndarray) -> np.ndarray:
        if (
            not isinstance(visible, np.ndarray)
            or visible.dtype != np.uint8
            or visible.ndim != 3
            or visible.shape[2] != 3
            or not visible.size
        ):
            raise ValueError("visible must be a nonempty uint8 RGB array")
        if (
            not isinstance(infrared, np.ndarray)
            or infrared.dtype != np.uint8
            or infrared.ndim != 2
            or infrared.shape != visible.shape[:2]
        ):
            raise ValueError("infrared must be an aligned uint8 grayscale array")
        fused = self.function(visible, infrared)
        if (
            not isinstance(fused, np.ndarray)
            or fused.dtype != np.uint8
            or fused.shape != visible.shape
        ):
            raise ValueError("fusion backend must return aligned uint8 RGB pixels")
        return fused


def load_fusion_backend(specification: str) -> CallableFusion:
    """Load an explicit user plugin, e.g. my_backend:fuse.

    The exported attribute must be a function or an initialized callable object,
    not a model class or a no-argument factory. No implicit model is selected.
    """
    module, separator, attribute = specification.partition(":")
    if not separator or not module or not attribute or ":" in attribute:
        raise ValueError("fusion plugin must be specified as module:callable")
    function = getattr(import_module(module), attribute)
    if isinstance(function, type):
        raise TypeError("export an initialized fusion object, not a class")
    return CallableFusion(function)
