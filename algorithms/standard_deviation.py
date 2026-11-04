
"""Algorithm 2: Standard Deviation Downsampling."""

from .common import (
    as_signal,
    moving_std,
    select_top_scores
)


def select_samples(
    x,
    keep,
    window=48,
    chunk=256
):
    """
    Select samples based on local signal variation.

    Higher standard deviation means higher
    priority for retention.
    """

    signal = as_signal(x)

    # Calculate local standard deviation.
    scores = moving_std(
        signal,
        window
    )

    # Select highest-priority samples.
    indices = select_top_scores(
        scores,
        keep,
        chunk
    )

    return indices