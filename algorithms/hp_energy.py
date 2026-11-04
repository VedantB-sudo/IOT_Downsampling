
"""Algorithm 4: High-Pass Energy Downsampling."""

from .common import (
    as_signal,
    high_pass,
    moving_mean,
    select_top_scores
)


def select_samples(
    x,
    keep,
    window=48,
    chunk=256,
    alpha=0.1
):
    """
    High-Pass Energy downsampling.

    Steps:
    1. Apply high-pass filtering.
    2. Calculate sliding-window energy.
    3. Prioritise high-energy regions.
    4. Select samples within the data budget.
    """

    signal = as_signal(x)

    # Step 1: High-pass filtering.
    filtered = high_pass(
        signal,
        alpha
    )

    # Step 2: Calculate mean-square energy.
    energy_scores = moving_mean(
        filtered ** 2,
        window
    )

    # Step 3: Select samples.
    indices = select_top_scores(
        energy_scores,
        keep,
        chunk
    )

    return indices