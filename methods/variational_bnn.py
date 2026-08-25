import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from torch import nn, optim

from src.losses import ELBOLoss
from src.metrics.ood import EntropyCriterion, MutualInformationCriterion
from src.training import ClassificationRoutine
from src.training.experiment import add_experiment_args, fit_and_evaluate, run_repeated
from src.utils import add_common_args, load_gpu_config, print_gpu_config

METHOD_NAME = "variational_bnn"


def get_bayesian_model(backbone, channels, classes):
    if backbone == "resnet":
        from src.models.resnet import bayesian_resnet1d
        return bayesian_resnet1d(in_channels=channels, num_classes=classes)
    if backbone == "lenet":
        from src.models.lenet import bayesian_lenet1d
        return bayesian_lenet1d(in_channels=channels, num_classes=classes)
    if backbone == "transformer":
        from src.models.transformer import bayesian_transformer1d
        return bayesian_transformer1d(in_channels=channels, num_classes=classes)
    raise ValueError(f"No Bayesian variant for backbone: {backbone}")


def run_once(args, seed, run_dir):
    def build(dm):
        model = get_bayesian_model(args.backbone, dm.num_channels, dm.num_classes)
        criterion = (MutualInformationCriterion() if args.ood_criterion == "mi"
                     else EntropyCriterion())
        return ClassificationRoutine(
            model=model, num_classes=dm.num_classes, is_ensemble=True,
            loss=ELBOLoss(model, nn.CrossEntropyLoss(), args.kl_weight, args.num_samples),
            optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
            eval_ood=True, eval_shift=args.eval_shift, ood_criterion=criterion,
        )
    return fit_and_evaluate(args, run_dir, build)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--ood-criterion", choices=["entropy", "mi"], default="mi")
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
