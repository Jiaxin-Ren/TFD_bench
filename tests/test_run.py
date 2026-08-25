"""Tests for the benchmark matrix runner."""

from __future__ import annotations

import argparse
import ast
import tempfile
import unittest
from pathlib import Path

import yaml

import run as benchmark_run
from methods.sghmc import planned_collection_size, validate_collection_schedule
from src.utils import add_common_args


class RunConfigTests(unittest.TestCase):
    def write_config(self, root: Path, payload: dict) -> Path:
        path = root / "experiment.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_builds_dataset_method_backbone_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "methods": ["max_softmax", "edl"],
                "backbones": ["resnet"],
                "datasets": [{"name": "seu", "root": "./data/SEU"}],
                "method_args": {"edl": {"reg_weight": 0.01}},
            }
            path = self.write_config(root, config)
            loaded = benchmark_run.load_config(path)
            commands = benchmark_run.build_commands(path, loaded)

        self.assertEqual(len(commands), 2)
        self.assertEqual(Path(commands[0][1]).stem, "max_softmax")
        self.assertEqual(Path(commands[1][1]).stem, "edl")
        self.assertIn("--reg-weight", commands[1])
        self.assertEqual(commands[1][commands[1].index("--reg-weight") + 1], "0.01")

    def test_rejects_incomplete_dataset(self) -> None:
        config = {
            "methods": ["max_softmax"],
            "backbones": ["resnet"],
            "datasets": [{"name": "seu"}],
        }
        with self.assertRaisesRegex(ValueError, "name and root"):
            benchmark_run.build_commands(Path("invalid.yaml"), config)

    def test_boolean_options_use_argparse_boolean_form(self) -> None:
        command: list[str] = []
        benchmark_run.append_option(command, "eval_noise", False)
        benchmark_run.append_option(command, "overwrite", True)
        self.assertEqual(command, ["--no-eval-noise", "--overwrite"])

    def test_shift_evaluation_config_and_cli_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_config(root, {"evaluation": {"shift": True}})
            parser = add_common_args(argparse.ArgumentParser(), str(path))

            self.assertTrue(parser.parse_args([]).eval_shift)
            self.assertFalse(parser.parse_args(["--no-eval-shift"]).eval_shift)
            self.assertTrue(parser.parse_args(["--eval-shift"]).eval_shift)

    def test_sghmc_collection_schedule_is_complete(self) -> None:
        args = argparse.Namespace(
            epochs=50,
            cycle_start=10,
            cycle_length=4,
            num_estimators=10,
        )
        self.assertEqual(planned_collection_size(**vars(args)), 10)
        self.assertEqual(validate_collection_schedule(args), 10)

    def test_sghmc_rejects_incomplete_collection_schedule(self) -> None:
        args = argparse.Namespace(
            epochs=50,
            cycle_start=20,
            cycle_length=5,
            num_estimators=10,
        )
        with self.assertRaisesRegex(ValueError, "only 7 estimators"):
            validate_collection_schedule(args)

    def test_shift_flag_is_forwarded_to_every_ood_evaluation_path(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        paths = list((project_root / "methods").glob("*.py"))
        paths.append(project_root / "src" / "training" / "experiment.py")

        missing = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                eval_ood = keywords.get("eval_ood")
                if not isinstance(eval_ood, ast.Constant) or eval_ood.value is not True:
                    continue
                eval_shift = keywords.get("eval_shift")
                correctly_forwarded = (
                    isinstance(eval_shift, ast.Attribute)
                    and isinstance(eval_shift.value, ast.Name)
                    and eval_shift.value.id == "args"
                    and eval_shift.attr == "eval_shift"
                )
                if not correctly_forwarded:
                    missing.append(f"{path.name}:{node.lineno}")

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
