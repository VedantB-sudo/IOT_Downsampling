
import numpy as np
import pandas as pd
from algorithms import METHODS, choose_samples
from intel_common import OUT, FEATURES, SIZE, load_data, transmit, score
from benchmark_intel_pid import run_feature

data = load_data()
results = []

for feature in FEATURES:
    windows, capacity, pid_counts, fixed_counts = run_feature(
        data, feature, measure=False
    )

    for method in METHODS:
        for mode, counts in [("PID", pid_counts), ("Fixed", fixed_counts)]:
            selected = []

            for x, count in zip(windows, counts):
                idx = np.sort(np.asarray(
                    choose_samples(x, method, count / SIZE), dtype=int
                ))
                if (len(idx) != count or len(np.unique(idx)) != count
                        or np.any(idx < 0) or np.any(idx >= SIZE)):
                    raise ValueError(f"Invalid selection: {method}")
                selected.append(idx)

            received, drops = transmit(selected, capacity)
            before = score(windows, selected)
            after = score(windows, received)

            results.append({
                "Sensor": feature, "Algorithm": method, "Mode": mode,
                "RMSE Before": before[0], "RMSE After": after[0],
                "PRDN Before": before[1], "PRDN After": after[1],
                "Events Before": before[2], "Events After": after[2],
                "Drops": drops
            })

OUT.mkdir(parents=True, exist_ok=True)
report = pd.DataFrame(results)
report.to_csv(OUT / "delivery_comparison.csv", index=False)

print(report.to_string(index=False))
print("\nSaved:", OUT / "delivery_comparison.csv")