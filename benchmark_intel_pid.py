
import numpy as np
import pandas as pd
from algorithms import METHODS, choose_samples
from intel_common import (
    OUT, FEATURES, SIZE, load_data, get_windows,
    make_schedule, network, score
)


def run_feature(df, feature, measure=True):
    windows = [
        x for _, _, x in get_windows(df, feature, minimum=SIZE)
    ]
    if not windows:
        raise ValueError(f"No complete windows for {feature}")

    capacity, pid_counts, fixed_counts = make_schedule(len(windows))
    print(f"{feature}: {len(windows)} complete windows")

    if not measure:
        return windows, capacity, pid_counts, fixed_counts

    rows = []
    for method in METHODS:
        for mode, counts in [("PID", pid_counts), ("Fixed", fixed_counts)]:
            selected = []
            for x, count in zip(windows, counts):
                idx = np.asarray(
                    choose_samples(x, method, count / SIZE), dtype=int
                )
                if (len(idx) != count or len(np.unique(idx)) != count
                        or np.any(idx < 0) or np.any(idx >= SIZE)):
                    raise ValueError(f"Invalid indices: {method}")
                selected.append(idx)

            rmse, prdn, events = score(windows, selected)
            drops, queue = network(counts, capacity)

            rows.append({
                "Sensor": feature, "Algorithm": method, "Mode": mode,
                "Retained %": 100 * sum(counts) / (len(counts) * SIZE),
                "RMSE": rmse, "PRDN %": prdn,
                "P95 event %": events, "Drops": drops,
                "Mean queue %": queue
            })
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    results = []

    for feature in FEATURES:
        results.extend(run_feature(data, feature))

    report = pd.DataFrame(results)
    report.to_csv(OUT / "pid_comparison.csv", index=False)

    print(report.to_string(index=False))
    print("\nSaved:", OUT / "pid_comparison.csv")