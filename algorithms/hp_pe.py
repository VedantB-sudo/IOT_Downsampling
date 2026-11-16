
from .common import as_signal, high_pass, permutation_entropy, select_top_scores


def select_samples(x, keep, window=48, chunk=256, alpha=0.1):
    signal = as_signal(x)
    filtered = high_pass(signal, alpha)
    entropy = permutation_entropy(filtered, window)
    return select_top_scores(entropy, keep, chunk)