
"""Connect individual algorithms to the benchmark."""

from . import (
    uniform_decimation,
    standard_deviation,
    hp_pe,
    hp_energy,
    hybrid
)


METHODS = [
    "uniform",
    "std_dev",
    "hp_pe",
    "hp_energy",
    "hybrid"
]


def choose_samples(
    x,
    method,
    keep,
    window=48,
    chunk=256,
    alpha=0.1,
    pe_weight=0.5
):

    if method == "uniform":
        return uniform_decimation.select_samples(
            x, keep
        )

    if method == "std_dev":
        return standard_deviation.select_samples(
            x, keep, window, chunk
        )

    if method == "hp_pe":
        return hp_pe.select_samples(
            x, keep, window, chunk, alpha
        )

    if method == "hp_energy":
        return hp_energy.select_samples(
            x, keep, window, chunk, alpha
        )

    if method == "hybrid":
        return hybrid.select_samples(
            x, keep, window, chunk,
            alpha, pe_weight
        )

    raise ValueError(
        f"Unknown algorithm: {method}"
    )