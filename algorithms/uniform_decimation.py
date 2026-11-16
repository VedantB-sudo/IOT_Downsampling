
import numpy as np
from .common import as_signal, retention_count


def select_samples(x, keep):
    signal = as_signal(x)
    n = len(signal)
    k = retention_count(n, keep)
    return np.rint(np.linspace(0, n - 1, k)).astype(int)