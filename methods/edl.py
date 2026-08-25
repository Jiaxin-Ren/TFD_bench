import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import torch.nn.functional as F
from torch import Tensor, nn, optim

from src.losses import DECLoss
from src.training import ClassificationRoutine
from src.training.experiment import add_experiment_args, fit_and_evaluate, run_repeated
from src.utils import add_common_args, get_model, load_gpu_config, print_gpu_config


METHOD_NAME = "edl"


class EvidenceWrapper(nn.Module):
    """Convert unconstrained backbone outputs into smooth non-negative evidence."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: Tensor) -> Tensor:
        return F.softplus(self.backbone(inputs))


class EDLRoutine(ClassificationRoutine):
    """Evaluate EDL predictions through the Dirichlet predictive mean."""

    def prediction_to_probs(self, evidence: Tensor) -> Tensor:
        alpha = evidence + 1.0
        return alpha / alpha.sum(dim=-1, keepdim=True)


def run_once(args, seed, run_dir):
    def build(dm):
        backbone = get_model(args.backbone, dm.num_channels, dm.num_classes)
        model = EvidenceWrapper(backbone)
        return EDLRoutine(
            model=model, num_classes=dm.num_classes,
            loss=DECLoss(reg_weight=args.reg_weight, loss_type=args.loss_type),
            optim_recipe=optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            eval_ood=True, eval_shift=args.eval_shift, ood_criterion="evidential",
        )
    return fit_and_evaluate(args, run_dir, build)


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--reg-weight", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--loss-type", choices=["digamma", "log", "mse"], default="digamma")
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
