
import numpy as np
import pandas as pd
from collections import deque
from algorithms import METHODS, choose_samples
from benchmark_pid import SENSORS, SIZE, LIMIT, FOLDER, run_sensor


def transmit(samples, capacity):
    queue = deque()
    received = [set() for _ in samples]
    drops = 0

    for w, (indices, available) in enumerate(zip(samples, capacity)):
        queue.extend((w, int(i)) for i in indices)

        for _ in range(min(available, len(queue))):
            window, index = queue.popleft()
            received[window].add(index)

        while len(queue) > LIMIT:
            queue.pop()
            drops += 1

    for window, index in queue:
        received[window].add(index)

    return received, drops


def evaluate(windows, samples):
    rmse_values, prdn_values, events = [], [], []

    for x, indices in zip(windows, samples):
        idx = np.array(sorted(indices), dtype=int)

        restored = (
            np.interp(np.arange(SIZE), idx, x[idx])
            if len(idx) else np.zeros_like(x)
        )

        rmse = np.sqrt(np.mean((x - restored) ** 2))
        changes = np.abs(np.diff(x))
        important = np.where(
            changes > np.percentile(changes, 95)
        )[0] + 1

        rmse_values.append(rmse)
        prdn_values.append(100 * rmse / max(x.std(), 1e-12))
        events.append(
            100 * np.isin(important, idx).mean()
            if len(important) else np.nan
        )

    return (
        np.mean(rmse_values),
        np.mean(prdn_values),
        np.nanmean(events)
    )


results = []

for sensor in SENSORS:
    windows, capacity, pid_counts, fixed_counts = run_sensor(
        sensor, measure=False
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
            before = evaluate(windows, selected)
            after = evaluate(windows, received)

            results.append([
                sensor, method, mode,
                round(before[0], 3), round(after[0], 3),
                round(before[1], 2), round(after[1], 2),
                round(before[2], 1), round(after[2], 1),
                drops
            ])

columns = [
    "Sensor", "Algorithm", "Mode",
    "RMSE Before", "RMSE After",
    "PRDN Before", "PRDN After",
    "Events Before", "Events After", "Drops"
]

report = pd.DataFrame(results, columns=columns)
report.to_csv(FOLDER / "delivery_comparison.csv", index=False)

print(report.to_string(index=False))
print("\nSaved: benchmark_results/pamap2/delivery_comparison.csv")