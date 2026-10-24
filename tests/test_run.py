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
from src.training.experiment import add_experiment_args
from src.utils import DEFAULT_CONFIG_PATH, add_common_args


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


class MethodParameterTests(unittest.TestCase):
    project_root = Path(__file__).resolve().parents[1]
    default_path = project_root / "configs" / "default.yaml"
    tuning_path = project_root / "configs" / "tuning.yaml"

    def parse_method(self, method: str, arguments: list[str]):
        """Execute the real script's parser setup without importing/running models."""
        path = self.project_root / "methods" / f"{method}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        main = next(
            node for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        statements = []
        for statement in main.body:
            if (
                isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "args"
                        for target in statement.targets)
            ):
                self.assertIsInstance(statement.value, ast.Call)
                self.assertIsInstance(statement.value.func, ast.Attribute)
                self.assertEqual(statement.value.func.value.id, "parser")
                self.assertEqual(statement.value.func.attr, "parse_args")
                statements.append(statement)
                break
            statements.append(statement)
        namespace = {
            "argparse": argparse,
            "add_common_args": add_common_args,
            "add_experiment_args": add_experiment_args,
        }
        with patch("sys.argv", [str(path), *arguments]):
            exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
            return namespace["args"]

    def test_default_entry_points_use_default_config(self) -> None:
        self.assertEqual(benchmark_run.DEFAULT_CONFIG, self.default_path)
        self.assertEqual(Path(DEFAULT_CONFIG_PATH), self.default_path)

    def test_default_and_tuning_configs_are_equivalent(self) -> None:
        self.assertEqual(
            benchmark_run.load_config(self.default_path),
            benchmark_run.load_config(self.tuning_path),
        )

    def test_all_method_fallbacks_match_tuning(self) -> None:
        config = benchmark_run.load_config(self.tuning_path)
        for method, parameters in config["method_args"].items():
            with self.subTest(method=method):
                args = self.parse_method(method, ["--dataset", "seu"])
                path = self.project_root / "methods" / f"{method}.py"
                tree = ast.parse(path.read_text(encoding="utf-8"))
                defaults = {}
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "add_argument"):
                        for keyword in node.keywords:
                            if keyword.arg == "default":
                                key = ast.literal_eval(node.args[0]).removeprefix("--").replace("-", "_")
                                defaults[key] = ast.literal_eval(keyword.value)
                for key, expected in parameters.items():
                    self.assertEqual(defaults[key], expected, f"{method}.{key}")
                    self.assertEqual(getattr(args, key), expected, f"{method}.{key}")

    def test_standalone_and_matrix_parameters_match_for_all_84_jobs(self) -> None:
        config = benchmark_run.load_config(self.tuning_path)
        commands = benchmark_run.build_commands(self.tuning_path, config)
        self.assertEqual(len(commands), 84)
        for command in commands:
            dataset, method, _ = benchmark_run.command_identity(command)
            with self.subTest(method=method, dataset=dataset):
                standalone = self.parse_method(method, [
                    "--config", str(self.tuning_path), "--dataset", dataset,
                ])
                matrix = self.parse_method(method, command[2:])
                self.assertEqual(vars(standalone), vars(matrix))
                self.assertEqual(standalone.epochs, 50)
                self.assertEqual(standalone.batch_size, 128)
                self.assertEqual(standalone.seeds, [0, 1, 2, 3, 4])

    def test_dataset_specific_parameters_and_roots(self) -> None:
        for method, dataset, parameter, expected in (
            ("edl", "seu", "loss_type", "mse"),
            ("edl", "xjtu", "loss_type", "digamma"),
            ("snapshot_ensemble", "seu", "cycle_length", 5),
            ("snapshot_ensemble", "pu", "cycle_length", 10),
            ("snapshot_ensemble", "xjtu", "cycle_length", 10),
            ("snapshot_ensemble", "cwru", "cycle_length", 10),
            ("sghmc", "seu", "friction", 0.05),
            ("sghmc", "xjtu", "friction", 0.1),
        ):
            with self.subTest(method=method, dataset=dataset):
                args = self.parse_method(method, ["--dataset", dataset])
                self.assertEqual(getattr(args, parameter), expected)
                self.assertEqual(args.data_root, f"/mnt/d/Data/Machine/{dataset.upper()}")

    def test_custom_config_and_explicit_cli_priority(self) -> None:
        payload = {
            "datasets": [
                {"name": "seu", "root": "./SEU"},
                {"name": "pu", "root": "./PU", "max_id_windows_per_file": 50,
                 "max_ood_windows_per_file": 20, "max_shift_windows_per_file": 30,
                 "split_mode": "temporal", "ood_subset": "near"},
            ],
            "training": {"epochs": 17, "batch_size": 7, "learning_rate": 0.002,
                         "n_runs": 2, "seeds": [11, 12], "val_split": 0.1},
            "evaluation": {"noise": False, "shift": False},
            "method_args": {"edl": {"reg_weight": 0.02, "loss_type": "mse"}},
            "method_dataset_args": {"edl": {"pu": {"loss_type": "log"}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            arguments = ["--config", str(path), "--dataset", "pu"]
            args = self.parse_method("edl", arguments)
            self.assertEqual((args.epochs, args.batch_size, args.lr), (17, 7, 0.002))
            self.assertEqual((args.n_runs, args.seeds, args.val_split), (2, [11, 12], 0.1))
            self.assertEqual((args.reg_weight, args.loss_type), (1e-5, "mse"))
            self.assertEqual(args.data_root, "./PU")
            self.assertEqual(args.max_id_windows_per_file, 50)
            self.assertEqual(args.max_ood_windows_per_file, 20)
            self.assertEqual(args.max_shift_windows_per_file, 30)
            self.assertEqual((args.split_mode, args.ood_subset), ("temporal", "near"))
            self.assertFalse(args.eval_noise)
            self.assertFalse(args.eval_shift)
            overridden = self.parse_method("edl", [
                *arguments, "--epochs", "3", "--data-root", "./manual",
                "--reg-weight", "0.7", "--loss-type", "digamma",
                "--eval-noise", "--eval-shift",
            ])
            self.assertEqual((overridden.epochs, overridden.data_root), (3, "./manual"))
            self.assertEqual((overridden.reg_weight, overridden.loss_type), (0.7, "digamma"))
            self.assertTrue(overridden.eval_noise)
            self.assertTrue(overridden.eval_shift)

    def test_boolean_cli_overrides_method_default(self) -> None:
        args = self.parse_method("mc_batch_norm", ["--no-last-only", "--num-estimators", "3"])
        self.assertFalse(args.last_only)
        self.assertEqual(args.num_estimators, 3)

    def test_yaml_method_settings_do_not_change_standalone_defaults(self) -> None:
        payload = {
            "method_args": {
                "edl": {"reg_weight": 0.7, "weight_decay": 0.1, "loss_type": "log"},
                "snapshot_ensemble": {"cycle_length": 13},
                "sghmc": {"friction": 0.9},
                "mc_dropout": {"num_estimators": 99, "dropout_rate": 0.8},
                "mc_batch_norm": {"last_only": False},
            },
            "method_dataset_args": {
                "edl": {"xjtu": {"loss_type": "mse"}},
                "snapshot_ensemble": {"pu": {"cycle_length": 17}},
                "sghmc": {"xjtu": {"friction": 0.8}},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            for method, dataset, expected in (
                ("edl", "seu", {"reg_weight": 1e-5, "weight_decay": 0.0, "loss_type": "mse"}),
                ("edl", "xjtu", {"loss_type": "digamma"}),
                ("snapshot_ensemble", "pu", {"cycle_length": 10}),
                ("sghmc", "xjtu", {"friction": 0.1}),
                ("mc_dropout", "seu", {"num_estimators": 10, "dropout_rate": 0.1}),
                ("mc_batch_norm", "seu", {"last_only": True}),
            ):
                with self.subTest(method=method, dataset=dataset):
                    args = self.parse_method(method, [
                        "--config", str(path), "--dataset", dataset,
                    ])
                    for key, value in expected.items():
                        self.assertEqual(getattr(args, key), value, f"{method}.{key}")

    def test_cli_overrides_dataset_specific_script_defaults(self) -> None:
        for method, dataset, option, value, parameter, expected in (
            ("edl", "xjtu", "--loss-type", "log", "loss_type", "log"),
            ("snapshot_ensemble", "pu", "--cycle-length", "7", "cycle_length", 7),
            ("sghmc", "xjtu", "--friction", "0.2", "friction", 0.2),
        ):
            with self.subTest(method=method, dataset=dataset):
                args = self.parse_method(method, ["--dataset", dataset, option, value])
                self.assertEqual(getattr(args, parameter), expected)


if __name__ == "__main__":
    unittest.main()
