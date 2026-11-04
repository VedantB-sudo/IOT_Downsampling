 
"""Algorithm 5: Hybrid HP-PE + HP-Energy."""

from .common import (
    as_signal,
    high_pass,
    permutation_entropy,
    moving_mean,
    scale_robust,
    select_top_scores
)


def select_samples(
    x,
    keep,
    window=48,
    chunk=256,
    alpha=0.1,
    pe_weight=0.5
):
    """
    Hybrid downsampling.

    Combines:
    - High-pass permutation entropy.
    - High-pass signal energy.

    Default weighting:
        50% HP-PE + 50% HP-Energy.
    """

    if not 0 <= pe_weight <= 1:
        raise ValueError(
            "pe_weight must be between 0 and 1"
        )

    signal = as_signal(x)

    # Step 1: High-pass filtering.
    filtered = high_pass(
        signal,
        alpha
    )

    # Step 2: Calculate permutation entropy.
    pe_scores = permutation_entropy(
        filtered,
        window
    )

    # Step 3: Calculate high-pass energy.
    energy_scores = moving_mean(
        filtered ** 2,
        window
    )

    # Step 4: Normalise both scores.
    pe_normalised = scale_robust(
        pe_scores
    )

    energy_normalised = scale_robust(
        energy_scores
    )

    # Step 5: Calculate weighted hybrid score.
    hybrid_scores = (
        pe_weight * pe_normalised
        + (1 - pe_weight) * energy_normalised
    )

    # Step 6: Select samples.
    indices = select_top_scores(
        hybrid_scores,
        keep,
        chunk
    )

    return indices