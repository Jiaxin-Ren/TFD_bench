import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn, optim

from src.metrics.ood import MutualInformationCriterion
from src.models.wrappers import BatchEnsemble
from src.training import ClassificationRoutine
from src.training.experiment import add_experiment_args, fit_and_evaluate, run_repeated
from src.training.transforms import RepeatTarget
from src.utils import add_common_args, get_model, load_gpu_config, print_gpu_config

METHOD_NAME = "batch_ensemble"


def run_once(args, seed, run_dir):
    def build(dm):
        base = get_model(args.backbone, dm.num_channels, dm.num_classes)
        model = BatchEnsemble(base, args.num_estimators, True, True)
        return ClassificationRoutine(
            model=model, num_classes=dm.num_classes, is_ensemble=True,
            format_batch_fn=RepeatTarget(args.num_estimators),
            loss=nn.CrossEntropyLoss(),
            optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
            eval_ood=True, eval_shift=args.eval_shift, ood_criterion=MutualInformationCriterion(),
        )
    return fit_and_evaluate(args, run_dir, build)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--num-estimators", type=int, default=4)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
