"""Smoke checks for the published runtime and numerical result pipeline."""

import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from analysis.collect_results import collect_from_results_dir, compute_summary_statistics
from analysis.generate_tables import generate_markdown_table
from src.post_processing import LaplaceApprox, TemperatureScaler
from src.training import ClassificationRoutine


class ProbabilityPostprocess(LaplaceApprox):
    """Exercise probability-valued post-processing without fitting a posterior."""

    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Identity()

    def forward(self, inputs):
        return inputs.softmax(dim=-1)


class PublicRuntimeTests(unittest.TestCase):
    def test_base_and_postprocessed_evaluation(self):
        logits = torch.tensor([[3., 1., 0.], [0., 2., 1.], [2., 1., 3.]])
        targets = torch.tensor([0, 1, 2])
        scaler = TemperatureScaler(nn.Identity(), init_val=2.)
        scaler.trained = True
        for postprocess in (None, scaler, ProbabilityPostprocess()):
            with self.subTest(postprocess=type(postprocess).__name__):
                routine = ClassificationRoutine(
                    nn.Identity(), num_classes=3, eval_ood=True, eval_shift=True,
                    post_processing=postprocess,
                    ood_criterion="msp" if postprocess is None else "post_processing",
                    collect_predictions=True,
                )
                routine._prediction_storage = {
                    key: [] for key in (
                        "id_probs", "id_base_probs", "id_targets", "id_ood_scores",
                        "ood_probs", "ood_base_probs", "ood_scores",
                    )
                }
                routine.test_step((logits, targets), 0, 0)
                routine.test_step((torch.zeros_like(logits), targets), 0, 1)
                routine.test_step((logits, targets), 0, 2)
                self.assertEqual(float(routine.test_cls_metrics.compute()["test/cls/Acc"]), 1.)
                self.assertEqual(float(routine.test_shift_metrics.compute()["shift/cls/Acc"]), 1.)
                artifacts = routine.get_prediction_artifacts()
                expected = logits / 2. if postprocess is scaler else logits
                torch.testing.assert_close(artifacts["id_probs"], expected.softmax(dim=-1))
                torch.testing.assert_close(artifacts["id_base_probs"], logits.softmax(dim=-1))
                self.assertEqual(artifacts["ood_scores"].shape, (3,))
                self.assertTrue(torch.isfinite(routine.test_ood_metrics.compute()["ood/AUROC"]))
                if postprocess is not None:
                    self.assertEqual(float(routine.post_cls_metrics.compute()["test/post/cls/Acc"]), 1.)
                    self.assertTrue(torch.isfinite(routine.post_sc_metrics.compute()["test/post/sc/AURC"]))

    def test_collection_and_table_preserve_seed_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pu" / "resnet" / "edl" / "runs.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["seed", "config", "test/cls/Acc"])
                writer.writeheader()
                for seed, accuracy in enumerate((.8, .9)):
                    writer.writerow({"seed": seed, "config": "clean", "test/cls/Acc": accuracy})
            summary = compute_summary_statistics(collect_from_results_dir(directory))
            stats = summary["pu/resnet/edl/clean"]["metrics"]["test/cls/Acc"]
            self.assertEqual(stats, {"mean": .85, "std": .0707, "n": 2})
            table = generate_markdown_table(summary, ["test/cls/Acc"], False)
            self.assertIn("85.00±7.07", table)


if __name__ == "__main__":
    unittest.main()
