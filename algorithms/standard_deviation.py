
from .common import as_signal, moving_std, select_top_scores


def select_samples(x, keep, window=48, chunk=256):
    signal = as_signal(x)
    scores = moving_std(signal, window)
    return select_top_scores(scores, keep, chunk)