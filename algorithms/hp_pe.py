
"""Algorithm 3: High-Pass Permutation Entropy."""

from .common import (
    as_signal,
    high_pass,
    permutation_entropy,
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
    High-Pass Permutation Entropy downsampling.

    Steps:
    1. Apply high-pass filter.
    2. Calculate permutation entropy.
    3. Prioritise complex signal regions.
    4. Select samples within the data budget.
    """

    signal = as_signal(x)

    # Step 1: High-pass filtering.
    filtered = high_pass(
        signal,
        alpha
    )

    # Step 2: Calculate local pattern complexity.
    entropy_scores = permutation_entropy(
        filtered,
        window
    )

    # Step 3: Select samples.
    indices = select_top_scores(
        entropy_scores,
        keep,
        chunk
    )

    return indices