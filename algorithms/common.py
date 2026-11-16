
import numpy as np
from itertools import permutations


def as_signal(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Signal must contain at least 3 valid readings.")
    return x


def retention_count(n, keep):
    if not 0 < keep <= 1:
        raise ValueError("Keep must be between 0 and 1.")
    return max(2, min(n, round(n * keep)))


def moving_mean(values, window):
    x = as_signal(values)
    if window < 1:
        raise ValueError("Window must be positive.")

    total = np.r_[0, np.cumsum(x)]
    end = np.arange(1, len(x) + 1)
    start = np.maximum(0, end - window)

    return (total[end] - total[start]) / (end - start)


def moving_std(values, window):
    x = as_signal(values)
    mean = moving_mean(x, window)
    variance = moving_mean(x ** 2, window) - mean ** 2
    return np.sqrt(np.maximum(variance, 0))


def high_pass(values, alpha=0.1):
    x = as_signal(values)
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1.")

    baseline = np.empty_like(x)
    baseline[0] = x[0]

    for i in range(1, len(x)):
        baseline[i] = alpha * x[i] + (1 - alpha) * baseline[i - 1]

    return x - baseline


def permutation_entropy(values, window=48):
    x = as_signal(values)
    if window < 3:
        raise ValueError("Window must be at least 3.")

    patterns = list(permutations(range(3)))
    pattern_ids = {pattern: i for i, pattern in enumerate(patterns)}

    codes = np.array([
        pattern_ids[tuple(np.argsort(x[i-2:i+1], kind="stable"))]
        for i in range(2, len(x))
    ])

    t = np.arange(len(x))
    end = np.maximum(0, t - 1)
    start = np.maximum(0, t - window - 1)

    counts = np.column_stack([
        np.r_[0, np.cumsum(codes == p)][end]
        - np.r_[0, np.cumsum(codes == p)][start]
        for p in range(6)
    ]).astype(float)

    total = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts, total, out=np.zeros_like(counts), where=total > 0
    )

    log_p = np.zeros_like(probabilities)
    np.log(probabilities, out=log_p, where=probabilities > 0)

    return -(probabilities * log_p).sum(axis=1) / np.log(6)


def scale_robust(scores):
    x = as_signal(scores)
    low, high = np.percentile(x, [5, 95])

    if high - low < 1e-12:
        return np.zeros_like(x)

    return np.clip((x - low) / (high - low), 0, 1)


def select_top_scores(scores, keep, chunk=256):
    x = as_signal(scores)

    if chunk < 2:
        raise ValueError("Chunk must be at least 2.")

    n = len(x)
    k = retention_count(n, keep)

    if k == n:
        return np.arange(n)

    blocks = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    sizes = np.array([b - a for a, b in blocks])

    ideal = sizes * k / n
    quota = np.floor(ideal).astype(int)

    remaining = k - quota.sum()
    order = np.argsort(-(ideal - quota), kind="stable")
    quota[order[:remaining]] += 1

    selected = set()

    for (a, b), count in zip(blocks, quota):
        best = np.argsort(-x[a:b], kind="stable")[:count]
        selected.update((a + best).tolist())

    for endpoint in (0, n - 1):
        if endpoint not in selected:
            removable = [i for i in selected if i not in (0, n - 1)]
            worst = min(removable, key=lambda i: (x[i], i))
            selected.remove(worst)
            selected.add(endpoint)

    indices = np.array(sorted(selected), dtype=int)
    assert len(indices) == k

    return indices


def select_spaced_scores(scores, keep):
    x = as_signal(scores)
    n = len(x)
    k = retention_count(n, keep)

    if k == n:
        return np.arange(n)

    positions = np.linspace(0, n - 1, k)
    boundaries = np.floor(
        (positions[:-1] + positions[1:]) / 2
    ).astype(int)

    selected = [0]

    for i in range(1, k - 1):
        start = boundaries[i - 1] + 1
        end = boundaries[i] + 1
        selected.append(start + np.argmax(x[start:end]))

    selected.append(n - 1)
    return np.array(selected, dtype=int)