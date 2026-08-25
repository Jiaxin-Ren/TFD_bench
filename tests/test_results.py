"""Tests for the public result schema and analysis loaders."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.collect_results import collect_from_results_dir, compute_summary_statistics
from analysis.generate_tables import _group_results, generate_markdown_table
from src.metrics import AURC, CovAt5Risk
from analysis.visualization.io import discover_prediction_runs, finite_ood_score_pair
from analysis.visualization.plot_all import (
    SELECTIVE_METRICS,
    _group_results as group_plot_results,
)
from analysis.visualization.risk_coverage import _risk_coverage
from src.training.experiment import _build_summary, _write_manifest, evaluate


class ResultSchemaTests(unittest.TestCase):
    def test_collects_runs_and_computes_seed_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method_dir = root / "demo" / "resnet" / "max_softmax"
            method_dir.mkdir(parents=True)
            with (method_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["seed", "config", "test/cls/Acc", "test/cal/ECE", "ood/AUROC"],
                )
                writer.writeheader()
                writer.writerows([
                    {"seed": 0, "config": "clean", "test/cls/Acc": 0.8, "test/cal/ECE": 0.1, "ood/AUROC": 0.7},
                    {"seed": 1, "config": "clean", "test/cls/Acc": 1.0, "test/cal/ECE": 0.2, "ood/AUROC": 0.9},
                ])

            records = collect_from_results_dir(str(root))
            summary = compute_summary_statistics(records)

        self.assertEqual(len(records), 2)
        stats = summary["demo/resnet/max_softmax/clean"]
        self.assertEqual(stats["n_runs"], 2)
        self.assertAlmostEqual(stats["metrics"]["test/cls/Acc"]["mean"], 0.9)
        self.assertEqual(stats["metrics"]["ood/AUROC"]["n"], 2)

    def test_operating_shift_is_collected_filtered_and_tabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method_dir = root / "demo" / "resnet" / "max_softmax"
            method_dir.mkdir(parents=True)
            with (method_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "seed", "config", "test/cls/Acc", "test/sc/Cov@5Risk"
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {"seed": 0, "config": "clean", "test/cls/Acc": 1.0},
                    {"seed": 0, "config": "operating_shift", "test/cls/Acc": 0.8},
                    {"seed": 0, "config": "gaussian_s1", "test/cls/Acc": 0.7},
                ])
            shift_records = collect_from_results_dir(str(root), test_config="operating_shift")
            summary = compute_summary_statistics(collect_from_results_dir(str(root)))

        self.assertEqual(len(shift_records), 1)
        self.assertEqual(shift_records[0]["config"], "operating_shift")
        self.assertEqual(shift_records[0]["metrics"]["test/sc/Cov@5Risk"], 0.0)
        configs = [group_key[2] for group_key, _ in _group_results(summary)]
        self.assertEqual(configs, ["clean", "operating_shift", "gaussian_s1"])
        table = generate_markdown_table(summary, ["test/cls/Acc"])
        self.assertIn("## demo / resnet / Operating shift", table)

    def test_temperature_baseline_is_not_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method_dir = root / "demo" / "resnet" / "temperature_scaling"
            method_dir.mkdir(parents=True)
            with (method_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["seed", "config", "test/cls/Acc", "test/cal/ECE"],
                )
                writer.writeheader()
                writer.writerows([
                    {"seed": 0, "config": "baseline_clean", "test/cls/Acc": 0.8, "test/cal/ECE": 0.2},
                    {"seed": 0, "config": "after_clean", "test/cls/Acc": 0.8, "test/cal/ECE": 0.05},
                ])
            records = collect_from_results_dir(str(root))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["method"], "temperature_scaling")
        self.assertEqual(records[0]["config"], "clean")
        self.assertAlmostEqual(records[0]["metrics"]["test/cal/ECE"], 0.05)

    def test_tidy_summary_and_manifest_contract(self) -> None:
        frame = pd.DataFrame([
            {"seed": 0, "config": "clean", "test/cls/Acc": 0.8},
            {"seed": 1, "config": "clean", "test/cls/Acc": 1.0},
            {"seed": 0, "config": "baseline_clean", "test/cls/Acc": 0.7},
        ])
        summary = _build_summary(frame)
        self.assertEqual(list(summary.columns), ["config", "metric", "mean", "std", "n"])
        self.assertEqual(int(summary.iloc[0]["n"]), 2)
        self.assertEqual(set(summary["config"]), {"clean"})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(dataset="demo", backbone="resnet", output_dir=str(root))
            _write_manifest(root, args, "max_softmax", [0, 1], status="complete")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["files"]["runs"], "runs.csv")

    def test_prediction_loader_ignores_legacy_object_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred_dir = root / "demo" / "resnet" / "max_softmax" / "seed0" / "predictions"
            pred_dir.mkdir(parents=True)
            np.savez(
                pred_dir / "clean.npz",
                id_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
                id_ood_scores=np.array([0.1, np.nan]),
                ood_scores=np.array([0.8, np.inf]),
                ood_criterion=np.array("legacy", dtype=object),
            )
            grouped = discover_prediction_runs(
                root, dataset="demo", backbone="resnet", config="clean"
            )
            arrays = grouped["max_softmax"][0][1]
            id_scores, ood_scores = finite_ood_score_pair(arrays)

        self.assertEqual(id_scores.tolist(), [0.1])
        self.assertEqual(ood_scores.tolist(), [0.8])
        self.assertNotIn("ood_criterion", arrays)


    def test_selective_metrics_use_native_uncertainty_ordering(self) -> None:
        probs = torch.tensor(
            [[0.6, 0.4], [0.9, 0.1], [0.7, 0.3], [0.8, 0.2]]
        )
        targets = torch.tensor([0, 1, 0, 1])
        native_uncertainty = torch.tensor([0.1, 0.9, 0.2, 0.8])

        msp_metric = AURC()
        msp_metric.update(probs, targets)
        native_metric = AURC()
        native_metric.update(probs, targets, native_uncertainty)

        self.assertLess(native_metric.compute().item(), msp_metric.compute().item())
    def test_risk_coverage_plot_uses_native_uncertainty(self) -> None:
        probs = np.array([[0.99, 0.01], [0.6, 0.4]])
        targets = np.array([1, 0])
        uncertainty = np.array([0.9, 0.1])
        _, risk, _ = _risk_coverage(probs, targets, uncertainty)
        self.assertEqual(risk.tolist(), [0.0, 0.5])

    def test_operating_shift_selective_metrics_are_grouped_for_plotting(self) -> None:
        metrics = {
            metric: {"mean": 0.1, "std": 0.0}
            for metric in SELECTIVE_METRICS
        }
        results = {
            "demo/resnet/max_softmax/operating_shift": {
                "dataset": "demo",
                "backbone": "resnet",
                "method": "max_softmax",
                "config": "operating_shift",
                "metrics": metrics,
            }
        }
        groups = group_plot_results(results, SELECTIVE_METRICS)
        key = ("demo", "resnet", "operating_shift")
        self.assertIn(key, groups)
        self.assertIn("Max Softmax", groups[key])


    def test_cov_at_5_risk_returns_largest_admissible_coverage(self) -> None:
        probs = torch.tensor(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]
        )
        targets = torch.tensor([0, 0, 1, 0])
        native_uncertainty = torch.tensor([0.1, 0.2, 0.3, 0.4])

        metric = CovAt5Risk()
        metric.update(probs, targets, native_uncertainty)

        self.assertAlmostEqual(metric.compute().item(), 0.5)
    def test_cov_at_5_risk_returns_zero_when_none_is_admissible(self) -> None:
        probs = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
        targets = torch.tensor([1, 1])
        uncertainty = torch.tensor([0.1, 0.2])
        metric = CovAt5Risk()
        metric.update(probs, targets, uncertainty)
        self.assertEqual(metric.compute().item(), 0.0)



    def test_operating_shift_is_saved_once_and_not_repeated_with_noise(self) -> None:
        class DummyRoutine:
            eval_shift = True
            eval_ood = True

            def get_prediction_artifacts(self):
                return {}

        class DummyDataModule:
            eval_shift = True
            eval_ood = True
            test = "clean"
            ood = "ood"
            shift = "operating_shift"
            noise_params = {"gaussian": [1, 2, 3, 4, 5]}
            ood_df = None

            def setup(self, stage):
                return None

            def get_shift_set(self):
                return self.shift

            def get_noisy_test_set(self, noise_type, severity):
                return f"{noise_type}_s{severity}"

            def get_noisy_ood_set(self, noise_type, severity):
                return f"{noise_type}_ood_s{severity}"

        class DummyTrainer:
            logger = None

            def __init__(self):
                self.calls = []

            def test(self, model, datamodule, ckpt_path=None):
                self.calls.append(
                    {
                        "test": datamodule.test,
                        "dm_shift": datamodule.eval_shift,
                        "routine_shift": model.eval_shift,
                        "eval_ood": datamodule.eval_ood and model.eval_ood,
                    }
                )
                return [{"test/cls/Acc": 1.0}]

        args = Namespace(eval_shift=True, eval_noise=True)
        trainer = DummyTrainer()
        routine = DummyRoutine()
        datamodule = DummyDataModule()
        configs = []

        results = evaluate(
            args,
            trainer,
            routine,
            datamodule,
            on_config=configs.append,
        )

        expected = ["clean", "operating_shift"] + [
            f"gaussian_s{severity}" for severity in range(1, 6)
        ]
        self.assertEqual(configs, expected)
        self.assertEqual(list(results), expected)
        self.assertEqual(
            sum(call["test"] == "operating_shift" for call in trainer.calls),
            1,
        )
        self.assertTrue(all(not call["dm_shift"] for call in trainer.calls))
        self.assertTrue(all(not call["routine_shift"] for call in trainer.calls))
        operating_call = trainer.calls[1]
        self.assertFalse(operating_call["eval_ood"])
        self.assertTrue(all(call["eval_ood"] for call in trainer.calls[2:]))
        self.assertTrue(datamodule.eval_shift)
        self.assertTrue(routine.eval_shift)


if __name__ == "__main__":
    unittest.main()