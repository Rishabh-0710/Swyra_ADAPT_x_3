"""
Memory accounting used to audit the bounded-memory guarantee.

``deep_nbytes`` walks an object graph (dicts, lists, dataclasses, sklearn
estimators and their tree predictors) and sums the bytes of every numpy array
plus the shallow size of Python containers. ``large_arrays`` lists every array
whose leading dimension exceeds a threshold, which the tests use to prove that
no historical batch is retained anywhere inside the learner.
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _children(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            yield f"[{i}]", v
    else:
        d = getattr(obj, "__dict__", None)
        if d is not None:
            for k, v in d.items():
                yield k, v
        for slot in getattr(type(obj), "__slots__", ()) or ():
            if hasattr(obj, slot):
                yield slot, getattr(obj, slot)


_SKIP_TYPES = (type, type(len), type(_children))


def deep_nbytes(obj: Any, _seen: Optional[Set[int]] = None) -> int:
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or isinstance(obj, _SKIP_TYPES) or callable(obj) and not hasattr(obj, "__dict__"):
        return 0
    _seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return int(obj.nbytes) + sum(deep_nbytes(x, _seen) for x in obj.ravel())
        return int(obj.nbytes)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return int(obj.memory_usage(deep=True).sum()) if isinstance(obj, pd.DataFrame) else int(obj.memory_usage(deep=True))
    size = sys.getsizeof(obj)
    for _, child in _children(obj):
        size += deep_nbytes(child, _seen)
    return size


def large_arrays(obj: Any, min_rows: int, path: str = "root", _seen: Optional[Set[int]] = None) -> List[Tuple[str, Tuple[int, ...]]]:
    """Return (path, shape) of all arrays/frames with more than ``min_rows`` rows."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or isinstance(obj, _SKIP_TYPES):
        return []
    _seen.add(id(obj))
    found: List[Tuple[str, Tuple[int, ...]]] = []
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        if obj.ndim >= 1 and obj.shape[0] > min_rows and not (isinstance(obj, np.ndarray) and obj.dtype.names):
            found.append((path, tuple(obj.shape)))
        return found
    for name, child in _children(obj):
        found.extend(large_arrays(child, min_rows, f"{path}.{name}", _seen))
    return found


def estimator_nbytes(estimator: Any) -> int:
    """Bytes held by a fitted histogram GBDT (tree node arrays + bin thresholds)."""
    return deep_nbytes(estimator)
