
"""Algorithm 1: Uniform Decimation."""

import numpy as np

from .common import as_signal, retention_count


def select_samples(x, keep):
    """
    Retain evenly spaced samples.

    x: Original sensor signal.
    keep: Fraction of samples to retain.

    Returns:
        Indices of retained samples.
    """

    signal = as_signal(x)

    n = len(signal)

    k = retention_count(n, keep)

    indices = np.rint(
        np.linspace(0, n - 1, k)
    ).astype(int)

    return indices