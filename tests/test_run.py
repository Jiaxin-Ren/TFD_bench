"""Tests for the benchmark matrix runner."""

from __future__ import annotations

import argparse
import ast
import csv
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_empty_method_dataset_args_preserves_existing_commands(self) -> None:
        config = {
            "methods": ["edl"],
            "backbones": ["resnet"],
            "datasets": [{"name": "seu", "root": "./data/SEU"}],
            "method_args": {"edl": {"reg_weight": 0.01}},
        }
        original = benchmark_run.build_commands(Path("experiment.yaml"), config)
        with_empty_section = benchmark_run.build_commands(
            Path("experiment.yaml"),
            {**config, "method_dataset_args": {}},
        )
        self.assertEqual(with_empty_section, original)

    def test_method_dataset_args_are_scoped_and_override_method_args(self) -> None:
        config = {
            "methods": ["edl", "mc_dropout"],
            "backbones": ["resnet"],
            "datasets": [
                {"name": "seu", "root": "./data/SEU"},
                {"name": "pu", "root": "./data/PU"},
            ],
            "method_args": {
                "edl": {"reg_weight": 0.01, "loss_type": "digamma"},
                "mc_dropout": {"num_estimators": 10},
            },
            "method_dataset_args": {
                "edl": {"seu": {"reg_weight": 0.02}},
                "mc_dropout": {"pu": {"dropout_rate": 0.2}},
            },
        }
        commands = benchmark_run.build_commands(Path("experiment.yaml"), config)
        indexed = {
            (
                Path(command[1]).stem,
                command[command.index("--dataset") + 1],
            ): command
            for command in commands
        }

        edl_seu = indexed[("edl", "seu")]
        edl_pu = indexed[("edl", "pu")]
        mc_seu = indexed[("mc_dropout", "seu")]
        mc_pu = indexed[("mc_dropout", "pu")]
        self.assertEqual(edl_seu[edl_seu.index("--reg-weight") + 1], "0.02")
        self.assertEqual(edl_pu[edl_pu.index("--reg-weight") + 1], "0.01")
        self.assertIn("--loss-type", edl_seu)
        self.assertNotIn("--dropout-rate", mc_seu)
        self.assertEqual(mc_pu[mc_pu.index("--dropout-rate") + 1], "0.2")
        self.assertNotIn("--dropout-rate", edl_pu)

    def test_method_dataset_args_reject_common_arguments(self) -> None:
        base = {
            "methods": ["edl"],
            "backbones": ["resnet"],
            "datasets": [{"name": "seu", "root": "./data/SEU"}],
        }
        for argument, value in (
            ("epochs", 10),
            ("lr", 0.01),
            ("batch-size", 32),
            ("seeds", [0]),
            ("val_split", 0.1),
            ("devices", 2),
            ("eval_noise", False),
            ("workers", 2),
        ):
            with self.subTest(argument=argument):
                config = {
                    **base,
                    "method_dataset_args": {
                        "edl": {"seu": {argument: value}},
                    },
                }
                with self.assertRaisesRegex(
                    ValueError,
                    rf"method_dataset_args\.edl\.seu.*common argument.*{argument}",
                ):
                    benchmark_run.build_commands(Path("experiment.yaml"), config)

    def test_method_dataset_args_reject_invalid_hierarchy(self) -> None:
        base = {
            "methods": ["edl"],
            "backbones": ["resnet"],
            "datasets": [{"name": "seu", "root": "./data/SEU"}],
        }
        invalid_values = (
            ([], "method_dataset_args must be a mapping"),
            ({"edl": []}, "method_dataset_args.edl must be a mapping"),
            (
                {"edl": {"seu": []}},
                "method_dataset_args.edl.seu must be a mapping",
            ),
        )
        for value, message in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, message):
                    benchmark_run.build_commands(
                        Path("experiment.yaml"),
                        {**base, "method_dataset_args": value},
                    )

    def test_forwards_dataset_specific_window_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "methods": ["max_softmax"],
                "backbones": ["resnet"],
                "datasets": [{
                    "name": "wt",
                    "root": "./data/WT",
                    "max_id_windows_per_file": 500,
                    "max_ood_windows_per_file": 100,
                    "max_shift_windows_per_file": 100,
                }],
            }
            path = self.write_config(root, config)
            command = benchmark_run.build_commands(path, config)[0]

        for option, expected in (
            ("--max-id-windows-per-file", "500"),
            ("--max-ood-windows-per-file", "100"),
            ("--max-shift-windows-per-file", "100"),
        ):
            self.assertEqual(command[command.index(option) + 1], expected)

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
        benchmark_run.append_option(command, "values", [1, 2])
        self.assertEqual(
            command,
            ["--no-eval-noise", "--overwrite", "--values", "1", "2"],
        )

    def test_parallel_gpu_parsing_and_single_gpu_overrides(self) -> None:
        self.assertEqual(benchmark_run.parse_gpu_ids("7,3,1", 3), [7, 3, 1])
        self.assertEqual(benchmark_run.parse_gpu_ids([4, 5], 2), [4, 5])
        with self.assertRaisesRegex(ValueError, "unique"):
            benchmark_run.parse_gpu_ids("0,0", 2)

        command = ["python", "method.py", "--devices", "8", "--strategy", "ddp"]
        overridden = benchmark_run.force_single_visible_gpu(command)
        self.assertEqual(
            overridden[-6:],
            ["--accelerator", "gpu", "--devices", "1", "--strategy", "auto"],
        )

    def test_resume_requires_complete_matching_result(self) -> None:
        command = [
            "python",
            str(benchmark_run.PROJECT_ROOT / "methods" / "max_softmax.py"),
            "--dataset", "demo",
            "--backbone", "resnet",
        ]
        with tempfile.TemporaryDirectory() as directory:
            config = {"output": {"dir": directory}}
            output = benchmark_run.result_dir(config, command)
            output.mkdir(parents=True)
            manifest = output / "manifest.json"
            manifest.write_text(
                '{"status":"complete","dataset":"demo",'
                '"method":"max_softmax","backbone":"resnet"}',
                encoding="utf-8",
            )
            self.assertFalse(benchmark_run.has_complete_result(config, command))

            (output / "runs.csv").write_text("seed,config\n", encoding="utf-8")
            (output / "summary.csv").write_text("config,metric\n", encoding="utf-8")
            self.assertTrue(benchmark_run.has_complete_result(config, command))

            manifest.write_text(
                '{"status":"complete","dataset":"other",'
                '"method":"max_softmax","backbone":"resnet"}',
                encoding="utf-8",
            )
            self.assertFalse(benchmark_run.has_complete_result(config, command))

    def test_parallel_runner_assigns_one_job_to_each_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {"output": {"dir": directory}}
            commands = [
                [
                    "python", "max_softmax.py", "--dataset", "seu",
                    "--backbone", "resnet",
                ],
                [
                    "python", "edl.py", "--dataset", "wt",
                    "--backbone", "resnet",
                ],
            ]
            barrier = threading.Barrier(2)

            def finish_together(*args, **kwargs):
                barrier.wait(timeout=2)
                return SimpleNamespace(returncode=0)

            with patch("run.subprocess.run", side_effect=finish_together) as mocked:
                failures = benchmark_run.run_parallel(
                    commands,
                    config,
                    gpu_ids=[2, 5],
                    continue_on_error=True,
                    resume=False,
                )

            with (Path(directory) / "runner_status.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertFalse(failures)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual({row["gpu"] for row in rows}, {"2", "5"})
        self.assertEqual({row["status"] for row in rows}, {"complete"})
        visible = {
            call.kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            for call in mocked.call_args_list
        }
        self.assertEqual(visible, {"2", "5"})
        for call in mocked.call_args_list:
            self.assertEqual(
                call.args[0][-6:],
                [
                    "--accelerator", "gpu", "--devices", "1",
                    "--strategy", "auto",
                ],
            )

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
