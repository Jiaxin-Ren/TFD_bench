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

# MSP is the deterministic reference used in every figure, but it is kept
# outside METHOD_ORDER so the paper's UQ-only result tables remain unchanged.
PLOT_METHOD_ORDER = ("max_softmax",) + METHOD_ORDER


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
PLOT_METHODS = frozenset(PLOT_METHOD_ORDER)
METHOD_RANK = {method: index for index, method in enumerate(METHOD_ORDER)}
PLOT_METHOD_RANK = {
    method: index for index, method in enumerate(PLOT_METHOD_ORDER)
}
DISPLAY_RANK = {
    METHOD_ABBREVIATIONS[method]: rank
    for method, rank in PLOT_METHOD_RANK.items()
}


def display_method(name: str) -> str:
    """Return the paper abbreviation for a method in the benchmark scope."""
    return METHOD_ABBREVIATIONS.get(name, name)



def method_display_sort_key(name: str) -> tuple[int, str]:
    """Sort displayed abbreviations in the paper's UQ-method order."""
    return DISPLAY_RANK.get(name, len(DISPLAY_RANK)), name

def method_sort_key(name: str) -> tuple[int, str]:
    """Sort methods in the same order as the paper's UQ-method table."""
    return METHOD_RANK.get(name, len(METHOD_RANK)), name


def plot_method_sort_key(name: str) -> tuple[int, str]:
    """Sort plotting methods with MSP first, followed by the paper order."""
    return PLOT_METHOD_RANK.get(name, len(PLOT_METHOD_RANK)), name

