"""Regression tests for stable, leakage-resistant dataset splits."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.mgb import MGBDataModule
from src.datasets.wt import ID_FILES, OOD_FILES, SHIFT_FILES
from src.datasets.utils import (
    assert_disjoint_sources,
    assert_disjoint_temporal_splits,
    temporal_window_split,
    subsample_uniform,
)


class DatasetSplitTests(unittest.TestCase):
    def test_temporal_split_is_stable_disjoint_and_gapped(self) -> None:
        data = [
            np.full((8, 1), index, dtype=np.float32)
            for index in range(100)
        ]
        labels = [2] * len(data)

        first = temporal_window_split(
            data, labels, "recording", val_ratio=0.2,
            test_ratio=0.2, gap_windows=1,
        )
        second = temporal_window_split(
            data, labels, "recording", val_ratio=0.2,
            test_ratio=0.2, gap_windows=1,
        )

        assert_disjoint_temporal_splits(*first)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)

        train, val, test = first
        self.assertLess(
            int(train["window_index"].max()) + 1,
            int(val["window_index"].min()),
        )
        self.assertLess(
            int(val["window_index"].max()) + 1,
            int(test["window_index"].min()),
        )

    def test_temporal_audit_rejects_reused_window(self) -> None:
        data = [np.array([[index]]) for index in range(20)]
        labels = [0] * len(data)
        train, val, test = temporal_window_split(
            data, labels, "recording", val_ratio=0.2,
            test_ratio=0.2, gap_windows=1,
        )
        leaked = pd.concat([val, train.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            assert_disjoint_temporal_splits(train, leaked, test)

    def test_file_level_audit_rejects_reused_source(self) -> None:
        train = pd.DataFrame({"source": ["train.csv"]})
        val = pd.DataFrame({"source": ["val.csv"]})
        test = pd.DataFrame({"source": ["train.csv"]})
        with self.assertRaisesRegex(RuntimeError, "Source-file leakage"):
            assert_disjoint_sources(train, val, test)

    def test_mgb_defaults_to_temporal_split(self) -> None:
        default = inspect.signature(MGBDataModule.__init__).parameters[
            "split_mode"
        ].default
        self.assertEqual(default, "temporal")

    def test_recording_datasets_do_not_randomly_mix_id_windows(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        for filename in ("pu.py", "cwru.py", "wt.py", "xjtu.py", "hit.py"):
            source = (
                project_root / "src" / "datasets" / filename
            ).read_text(encoding="utf-8")
            self.assertIn("build_temporal_id_splits", source, filename)
            self.assertNotIn(
                "id_df.sample(frac=1", source,
                f"{filename} randomly mixes windows from one recording",
            )
            self.assertIn("_split_done", source, filename)

    def test_uniform_limit_is_stable_and_covers_the_recording(self) -> None:
        data = list(range(1000))
        labels = [3] * len(data)
        first_data, first_labels = subsample_uniform(data, labels, 100)
        second_data, second_labels = subsample_uniform(data, labels, 100)

        self.assertEqual(first_data, second_data)
        self.assertEqual(first_labels, second_labels)
        self.assertEqual(len(first_data), 100)
        self.assertEqual(first_data[0], 0)
        self.assertEqual(first_data[-1], 999)
        self.assertEqual(len(set(first_data)), 100)

    def test_wt_uses_one_assembly_and_isolates_speed_shift(self) -> None:
        self.assertEqual(len(ID_FILES), 9)
        self.assertEqual(len(SHIFT_FILES), 12)
        self.assertEqual(len(OOD_FILES), 6)
        for file_list in (ID_FILES, SHIFT_FILES, OOD_FILES):
            assemblies = {
                int(filename.split("_")[0][-1])
                for _, filename, _ in file_list
            }
            self.assertEqual(assemblies, {1})



if __name__ == "__main__":
    unittest.main()
