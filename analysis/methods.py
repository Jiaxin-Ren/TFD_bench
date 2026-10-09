"""Canonical method scope, abbreviations, and ordering for benchmark analysis."""

METHOD_ORDER = (
    "temperature_scaling",
    "variational_bnn",
    "sgld",
    "sghmc",
    "deep_ensemble",
    "snapshot_ensemble",
    "batch_ensemble",
    "packed_ensemble",
    "swag",
    "laplace_approx",
    "mc_dropout",
    "mc_batch_norm",
    "edl",
)

# MSP is the deterministic reference retained in collected results, but kept
# outside METHOD_ORDER so the paper's UQ-only result tables remain unchanged.
RESULT_METHODS = frozenset(("max_softmax",) + METHOD_ORDER)


METHOD_ABBREVIATIONS = {
    "max_softmax": "MSP",
    "temperature_scaling": "TS",
    "variational_bnn": "VBNN",
    "sgld": "SGLD",
    "sghmc": "SGHMC",
    "deep_ensemble": "DE",
    "snapshot_ensemble": "SE",
    "batch_ensemble": "BE",
    "packed_ensemble": "PE",
    "swag": "SWAG",
    "laplace_approx": "LA",
    "mc_dropout": "MCD",
    "mc_batch_norm": "MCBN",
    "edl": "EDL",
}

BENCHMARK_METHODS = frozenset(METHOD_ORDER)
METHOD_RANK = {method: index for index, method in enumerate(METHOD_ORDER)}


def display_method(name: str) -> str:
    """Return the paper abbreviation for a method in the benchmark scope."""
    return METHOD_ABBREVIATIONS.get(name, name)



def method_sort_key(name: str) -> tuple[int, str]:
    """Sort methods in the same order as the paper's UQ-method table."""
    return METHOD_RANK.get(name, len(METHOD_RANK)), name


