
"""Shared functions for IoT downsampling algorithms."""

from itertools import permutations
import numpy as np


def as_signal(values):
    x = np.asarray(values, dtype=float)

    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError(
            "Signal must be a finite 1-D array "
            "with at least 3 readings."
        )

    return x


def retention_count(n, keep):
    if not 0 < keep <= 1:
        raise ValueError("keep must be in (0, 1].")

    return max(2, min(n, round(n * keep)))


def moving_mean(values, window):
    if window < 1:
        raise ValueError("window must be positive")

    x = as_signal(values)

    cumulative = np.concatenate(([0.0], np.cumsum(x)))
    ends = np.arange(1, len(x) + 1)
    starts = np.maximum(0, ends - window)

    return (
        (cumulative[ends] - cumulative[starts])
        / (ends - starts)
    )


def moving_std(values, window):
    x = as_signal(values)
    avg = moving_mean(x, window)

    variance = np.maximum(
        0.0,
        moving_mean(x * x, window) - avg * avg
    )

    return np.sqrt(variance)


def high_pass(values, alpha=0.1):
    """EMA-based high-pass filter approximation."""

    x = as_signal(values)

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")

    baseline = np.empty(len(x), dtype=float)
    baseline[0] = x[0]

    for i in range(1, len(x)):
        baseline[i] = (
            alpha * x[i]
            + (1 - alpha) * baseline[i - 1]
        )

    return x - baseline


def permutation_entropy(values, window):
    """Rolling permutation entropy with embedding dimension 3."""

    x = as_signal(values)

    if window < 3:
        raise ValueError("window must be at least 3")

    pattern_ids = {
        p: j
        for j, p in enumerate(permutations(range(3)))
    }

    codes = np.fromiter(
        (
            pattern_ids[
                tuple(
                    np.argsort(
                        x[i - 2:i + 1],
                        kind="stable"
                    )
                )
            ]
            for i in range(2, len(x))
        ),
        dtype=np.int64,
        count=len(x) - 2
    )

    t = np.arange(len(x))
    ends = np.maximum(0, t - 1)
    starts = np.maximum(0, t - window - 1)

    counts = []

    for pattern in range(6):
        cumulative = np.concatenate((
            [0],
            np.cumsum(codes == pattern)
        ))

        counts.append(
            cumulative[ends] - cumulative[starts]
        )

    counts = np.stack(
        counts, axis=1
    ).astype(float)

    totals = counts.sum(axis=1, keepdims=True)

    probabilities = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts),
        where=totals > 0
    )

    log_probabilities = np.zeros_like(probabilities)

    np.log(
        probabilities,
        out=log_probabilities,
        where=probabilities > 0
    )

    return (
        -(probabilities * log_probabilities)
        .sum(axis=1) / np.log(6)
    )


def scale_robust(scores):
    """Normalise scores to the range 0 to 1."""

    values = as_signal(scores)

    low, high = np.quantile(
        values, [0.05, 0.95]
    )

    if high - low < 1e-12:
        return np.zeros_like(values)

    return np.clip(
        (values - low) / (high - low),
        0,
        1
    )


def select_top_scores(scores, keep, chunk=256):
    """Select exactly the target number of samples."""

    values = as_signal(scores)

    if chunk < 2:
        raise ValueError("chunk must be at least 2")

    n = len(values)
    k = retention_count(n, keep)

    if k == n:
        return np.arange(n, dtype=int)

    blocks = [
        (a, min(a + chunk, n))
        for a in range(0, n, chunk)
    ]

    capacity = np.array([
        b - a for a, b in blocks
    ])

    ideal = capacity * (k / n)
    quota = np.floor(ideal).astype(int)

    remainder = k - int(quota.sum())

    for j in np.argsort(
        -(ideal - quota),
        kind="stable"
    )[:remainder]:
        quota[j] += 1

    selected = set()

    for (a, b), count in zip(blocks, quota):
        if count:
            best = np.argsort(
                -values[a:b],
                kind="stable"
            )[:count]

            selected.update(
                (a + best).tolist()
            )

    # Preserve first and last samples.
    for required in (0, n - 1):
        if required not in selected:
            removable = [
                i for i in selected
                if i not in (0, n - 1)
            ]

            worst = min(
                removable,
                key=lambda i: (values[i], i)
            )

            selected.remove(worst)
            selected.add(required)

    result = np.array(
        sorted(selected),
        dtype=int
    )

    assert len(result) == k

    return result