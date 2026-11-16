
from .common import as_signal, high_pass, moving_mean, select_top_scores


def select_samples(x, keep, window=48, chunk=256, alpha=0.1):
    signal = as_signal(x)
    filtered = high_pass(signal, alpha)
    energy = moving_mean(filtered ** 2, window)
    return select_top_scores(energy, keep, chunk)