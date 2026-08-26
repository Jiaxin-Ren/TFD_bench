"""Tests for the shared persistent dataset cache."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.datasets.cache import cache_path, load_cached_array


class DatasetCacheTests(unittest.TestCase):
    def test_second_load_reuses_cached_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "signal.txt"
            source.write_text("1\n2\n3\n", encoding="utf-8")
            calls = 0

            def loader(path: Path) -> np.ndarray:
                nonlocal calls
                calls += 1
                return np.loadtxt(path).reshape(-1, 1)

            first = load_cached_array(
                source, "demo", loader, cache_root=root / "cache"
            )
            second = load_cached_array(
                source, "demo", loader, cache_root=root / "cache"
            )

            self.assertEqual(calls, 1)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.dtype, np.float32)

    def test_source_or_parameters_change_the_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "signal.txt"
            source.write_text("1\n", encoding="utf-8")
            cache_root = root / "cache"
            initial = cache_path(
                source, "demo", {"channel": 0}, cache_root=cache_root
            )
            changed_parameters = cache_path(
                source, "demo", {"channel": 1}, cache_root=cache_root
            )

            source.write_text("1\n2\n", encoding="utf-8")
            os.utime(source, None)
            changed_source = cache_path(
                source, "demo", {"channel": 0}, cache_root=cache_root
            )

            self.assertNotEqual(initial, changed_parameters)
            self.assertNotEqual(initial, changed_source)

    def test_corrupt_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "signal.txt"
            source.write_text("1\n2\n", encoding="utf-8")
            cache_root = root / "cache"
            target = cache_path(source, "demo", cache_root=cache_root)
            target.parent.mkdir(parents=True)
            target.write_bytes(b"not a numpy file")
            calls = 0

            def loader(path: Path) -> np.ndarray:
                nonlocal calls
                calls += 1
                return np.loadtxt(path).reshape(-1, 1)

            result = load_cached_array(
                source, "demo", loader, cache_root=cache_root
            )

            self.assertEqual(calls, 1)
            np.testing.assert_array_equal(
                result, np.array([[1.0], [2.0]], dtype=np.float32)
            )
            np.testing.assert_array_equal(
                np.load(target, allow_pickle=False), result
            )


if __name__ == "__main__":
    unittest.main()
