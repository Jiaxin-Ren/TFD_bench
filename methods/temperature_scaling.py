import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn

from src.post_processing import TemperatureScaler
from src.training import ClassificationRoutine
from src.training.experiment import (
    add_experiment_args,
    evaluate,
    promote_postprocess_metrics,
    run_repeated,
    train_base_classifier,
)
from src.utils import add_common_args, load_gpu_config, print_gpu_config

METHOD_NAME = "temperature_scaling"


def run_once(args, seed, run_dir):
    trainer, datamodule, model = train_base_classifier(args, run_dir)
    raw_routine = ClassificationRoutine(
        model=model, num_classes=datamodule.num_classes,
        loss=nn.CrossEntropyLoss(), eval_ood=True, eval_shift=args.eval_shift,
    )
    baseline = evaluate(args, trainer, raw_routine, datamodule, artifact_prefix="baseline_")

    scaler = TemperatureScaler(
        model=model,
        device=next(model.parameters()).device,
        min_temperature=args.min_temperature,
        max_temperature=args.max_temperature,
    )
    scaler.fit(datamodule.postprocess_dataloader())
    temperature = scaler.temperature[0].item()
    scaler.fit = lambda *args, **kwargs: None
    scaled_routine = ClassificationRoutine(
        model=model, num_classes=datamodule.num_classes,
        loss=nn.CrossEntropyLoss(), eval_ood=True, eval_shift=args.eval_shift,
        post_processing=scaler, log_post_processing=True,
        ood_criterion="post_processing",
    )
    after = promote_postprocess_metrics(
        evaluate(args, trainer, scaled_routine, datamodule, artifact_prefix="after_")
    )
    results = {f"baseline_{name}": metrics for name, metrics in baseline.items()}
    results.update({
        f"after_{name}": {**metrics, "temperature": temperature}
        for name, metrics in after.items()
    })
    return results


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--min-temperature", type=float, default=0.1)
    parser.add_argument("--max-temperature", type=float, default=10.0)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
