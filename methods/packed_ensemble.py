import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn, optim

from src.metrics.ood import MutualInformationCriterion
from src.training import ClassificationRoutine
from src.training.experiment import add_experiment_args, fit_and_evaluate, run_repeated
from src.training.transforms import RepeatTarget
from src.utils import add_common_args, load_gpu_config, print_gpu_config

METHOD_NAME = "packed_ensemble"


def get_packed_model(backbone, channels, classes, estimators, alpha, gamma):
    if backbone == "resnet":
        from src.models.resnet import packed_resnet1d
        return packed_resnet1d(
            in_channels=channels, num_classes=classes, num_estimators=estimators,
            alpha=alpha, gamma=gamma,
        )
    if backbone == "lenet":
        from src.models.lenet import packed_lenet1d
        return packed_lenet1d(
            in_channels=channels, num_classes=classes, num_estimators=estimators,
            alpha=alpha, gamma=gamma,
        )
    if backbone == "transformer":
        from src.models.transformer import packed_transformer1d
        return packed_transformer1d(
            in_channels=channels, num_classes=classes, num_estimators=estimators,
            alpha=alpha, gamma=gamma,
        )
    raise ValueError(f"No packed variant for backbone: {backbone}")


def run_once(args, seed, run_dir):
    def build(dm):
        model = get_packed_model(
            args.backbone, dm.num_channels, dm.num_classes,
            args.num_estimators, args.alpha, args.gamma,
        )
        return ClassificationRoutine(
            model=model, num_classes=dm.num_classes, is_ensemble=True,
            format_batch_fn=RepeatTarget(args.num_estimators), loss=nn.CrossEntropyLoss(),
            optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
            eval_ood=True, eval_shift=args.eval_shift, ood_criterion=MutualInformationCriterion(),
        )
    return fit_and_evaluate(args, run_dir, build)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--num-estimators", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4)
    parser.add_argument("--gamma", type=int, default=1)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
