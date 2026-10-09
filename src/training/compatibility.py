"""Explicit method/backbone support policy for protocol-v2 experiments."""

from __future__ import annotations

from dataclasses import dataclass


BACKBONES = ("resnet", "lenet", "mlp", "transformer", "lstm", "timesnet")


@dataclass(frozen=True)
class Compatibility:
    supported: bool
    reason: str = ""


_SUPPORTED = {
    "variational_bnn": {"resnet", "lenet", "transformer"},
    "packed_ensemble": {"resnet", "lenet"},
    "mc_batch_norm": {"resnet", "transformer", "lstm"},
}

_REASONS = {
    ("packed_ensemble", "transformer"): (
        "Packed Transformer is excluded because its current packed forward pass is "
        "shape-incompatible with the 1D benchmark input."
    ),
}


def method_backbone_compatibility(method: str, backbone: str) -> Compatibility:
    """Return the declared support status; ordinary methods support all backbones."""
    if backbone not in BACKBONES:
        return Compatibility(False, f"Unknown backbone: {backbone}")
    allowed = _SUPPORTED.get(method)
    if allowed is None or backbone in allowed:
        return Compatibility(True)
    reason = _REASONS.get(
        (method, backbone),
        f"{method} has no current {backbone} implementation in protocol v2.",
    )
    return Compatibility(False, reason)

