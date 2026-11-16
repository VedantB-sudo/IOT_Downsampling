
from .common import as_signal, high_pass, permutation_entropy, moving_mean, scale_robust, select_top_scores


def select_samples(x, keep, window=48, chunk=256, alpha=0.1, pe_weight=0.5):
    if not 0 <= pe_weight <= 1:
        raise ValueError("pe_weight must be between 0 and 1")

    filtered = high_pass(as_signal(x), alpha)
    pe = scale_robust(permutation_entropy(filtered, window))
    energy = scale_robust(moving_mean(filtered ** 2, window))
    scores = pe_weight * pe + (1 - pe_weight) * energy

    return select_top_scores(scores, keep, chunk)