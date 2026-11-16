
import numpy as np


def evaluate_fidelity(original, indices, values=None):
    x = np.asarray(original, dtype=float)
    ix = np.asarray(indices, dtype=int)

    if values is None:
        values = x[ix]

    y = np.asarray(values, dtype=float)

    if len(x) < 3 or len(ix) < 2 or len(ix) != len(y):
        raise ValueError("Invalid signal or retained samples")

    if np.any(ix < 0) or np.any(ix >= len(x)) or len(np.unique(ix)) != len(ix):
        raise ValueError("Invalid sample indices")

    order = np.argsort(ix)
    ix, y = ix[order], y[order]

    reconstructed = np.interp(np.arange(len(x)), ix, y)
    error = x - reconstructed

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    std = float(np.std(x))
    prdn = 100 * rmse / std if std > 1e-12 else np.nan

    changes = np.abs(np.diff(x))
    results = {"RMSE": rmse, "MAE": mae, "PRDN_pct": prdn}

    for p in (95, 99):
        threshold = np.percentile(changes, p)
        important = np.flatnonzero(changes > threshold) + 1
        results[f"P{p}_retained_pct"] = (
            100 * np.isin(important, ix).mean() if len(important) else np.nan
        )

    return results