# SPDX-License-Identifier: Apache-2.0
"""Shared, persistent cache for parsed dataset signals."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


CACHE_SCHEMA_VERSION = 1
CACHE_ENV_VAR = "TFD_DATA_CACHE"


def data_cache_root() -> Path:
    """Return the configurable root used by all dataset signal caches."""
    configured = os.environ.get(CACHE_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "tfd_bench" / "datasets"


def cache_path(
    source: str | Path,
    dataset: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    """Build a cache path that changes with the source file and preprocessing."""
    source = Path(source).expanduser().resolve(strict=True)
    stat = source.stat()
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "source": str(source),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "parameters": dict(parameters or {}),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem).strip("._")
    root = Path(cache_root).expanduser() if cache_root else data_cache_root()
    return root / dataset.lower() / f"{safe_stem}_{digest}.npy"


def load_cached_array(
    source: str | Path,
    dataset: str,
    loader: Callable[[Path], np.ndarray],
    *,
    parameters: Mapping[str, Any] | None = None,
    dtype: np.dtype | type | None = np.float32,
    cache_root: str | Path | None = None,
) -> np.ndarray:
    """Load a numeric array from cache, or parse and cache it atomically.

    The cache stores only deterministic per-file preprocessing. Windowing,
    labels, data splits, augmentation, and normalization remain outside it.
    """
    target = cache_path(
        source, dataset, parameters=parameters, cache_root=cache_root
    )
    if target.is_file():
        try:
            cached = np.load(target, allow_pickle=False)
            if cached.dtype == object:
                raise ValueError("object arrays are not valid signal caches")
            return cached
        except (OSError, ValueError, EOFError):
            pass

    array = np.asarray(loader(Path(source)))
    if array.dtype == object:
        raise TypeError(f"Dataset cache only supports numeric arrays: {source}")
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    array = np.ascontiguousarray(array)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}_",
            suffix=".npy",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return array
