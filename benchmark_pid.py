
import numpy as np
import pandas as pd
from pathlib import Path
from algorithms import METHODS, choose_samples

SIZE, LIMIT = 1000, 1200
PATTERN = [450] * 4 + [60] * 8 + [450] * 10
FOLDER = Path("benchmark_results/pamap2")

SENSORS = pd.read_csv(
    FOLDER / "full_dataset/benchmark_summary.csv"
)["sensor_type"].dropna().unique().tolist()


def sensor_column(name):
    parts = name.lower().split("_")
    starts = {"hand": 3, "chest": 20, "ankle": 37}
    offsets = {"temp": 0, "acc": 1, "acc16": 1,
               "acc6": 4, "gyro": 7, "mag": 10}
    axes = {"x": 0, "y": 1, "z": 2}

    if len(parts) < 2 or parts[0] not in starts or parts[1] not in offsets:
        raise ValueError(f"Unknown sensor: {name}")

    axis = 0
    if parts[1] != "temp":
        if len(parts) != 3 or parts[2] not in axes:
            raise ValueError(f"Unknown sensor axis: {name}")
        axis = axes[parts[2]]

    return starts[parts[0]] + offsets[parts[1]] + axis


class PID:
    def __init__(self):
        self.keep, self.integral, self.previous = 0.30, 0, 0

    def update(self, queue):
        error = 0.25 - queue
        derivative = error - self.previous
        new_integral = np.clip(self.integral + error, -2, 2)

        trial = 0.30 + 0.55 * error + 0.05 * new_integral + 0.10 * derivative

        if (0.05 <= trial <= 0.50 or
                (trial > 0.50 and error < 0) or
                (trial < 0.05 and error > 0)):
            self.integral = new_integral

        target = 0.30 + 0.55 * error + 0.05 * self.integral + 0.10 * derivative
        self.keep = np.clip(
            self.keep + np.clip(target - self.keep, -0.08, 0.08),
            0.05, 0.50
        )
        self.previous = error
        return self.keep


def load_windows(sensor):
    pools = {}
    column = sensor_column(sensor)

    files = sorted(
        p for p in Path(".").rglob("subject*.dat")
        if "protocol" in [s.lower() for s in p.parts]
    )

    if not files:
        raise FileNotFoundError("PAMAP2 Protocol dataset not found")

    for path in files:
        data = pd.read_csv(
            path, sep=r"\s+", header=None,
            usecols=[0, 1, column]
        )
        data = data[(data[1] > 0) & data[column].notna()]

        groups = (
            data[1].ne(data[1].shift()) |
            data.index.to_series().diff().ne(1) |
            data[0].diff().gt(0.025)
        ).cumsum()

        for _, part in data.groupby(groups):
            x = part[column].to_numpy()
            key = (path.stem, int(part[1].iloc[0]))

            for i in range(0, len(x) - SIZE + 1, SIZE):
                pools.setdefault(key, []).append(x[i:i + SIZE])

    rng = np.random.default_rng(42)
    keys = list(pools)
    rng.shuffle(keys)

    for key in keys:
        rng.shuffle(pools[key])

    windows = []
    while any(pools.values()):
        for key in keys:
            if pools[key]:
                windows.append(pools[key].pop())

    if not windows:
        raise ValueError(f"No valid windows for {sensor}")

    print(f"{sensor}: {len(windows)} windows")
    return windows


def network(counts, capacity):
    queue = drops = 0
    occupancy = []

    for count, available in zip(counts, capacity):
        queue = max(0, queue + count - available)
        drops += max(0, queue - LIMIT)
        queue = min(queue, LIMIT)
        occupancy.append(queue / LIMIT)

    return drops, np.mean(occupancy)


def scores(windows, method, counts):
    rmse_values, prdn_values, events = [], [], []

    for x, count in zip(windows, counts):
        idx = np.sort(np.asarray(
            choose_samples(x, method, count / SIZE), dtype=int
        ))

        if (len(idx) != count or len(np.unique(idx)) != count
                or np.any(idx < 0) or np.any(idx >= SIZE)):
            raise ValueError(f"Invalid selection: {method}")

        restored = np.interp(np.arange(SIZE), idx, x[idx])
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

    return np.mean(rmse_values), np.mean(prdn_values), np.nanmean(events)


def run_sensor(sensor, measure=True):
    windows = load_windows(sensor)
    rng = np.random.default_rng(42)

    capacity = (PATTERN * (
        (len(windows) + len(PATTERN) - 1) // len(PATTERN)
    ))[:len(windows)]

    capacity = [
        max(10, int(c * rng.uniform(0.8, 1.2)))
        for c in capacity
    ]

    pid = PID()
    queue = 0
    pid_counts = []

    for available in capacity:
        count = round(SIZE * pid.update(queue / LIMIT))
        pid_counts.append(count)
        queue = min(LIMIT, max(0, queue + count - available))

    base, remainder = divmod(sum(pid_counts), len(pid_counts))
    fixed_counts = [
        base + ((i + 1) * remainder // len(pid_counts)
                - i * remainder // len(pid_counts))
        for i in range(len(pid_counts))
    ]

    if not measure:
        return windows, capacity, pid_counts, fixed_counts

    rows = []
    for method in METHODS:
        for mode, counts in [("PID", pid_counts), ("Fixed", fixed_counts)]:
            rmse, prdn, events = scores(windows, method, counts)
            drops, mean_queue = network(counts, capacity)

            rows.append([
                sensor, method, mode,
                round(100 * sum(counts) / (len(counts) * SIZE), 1),
                round(rmse, 4), round(prdn, 2),
                round(events, 1), drops,
                round(mean_queue * 100, 1)
            ])

    return rows


if __name__ == "__main__":
    results = []

    for sensor in SENSORS:
        results.extend(run_sensor(sensor))

    columns = [
        "Sensor", "Algorithm", "Mode", "Retained %",
        "RMSE", "PRDN %", "P95 event %", "Drops", "Mean queue %"
    ]

    report = pd.DataFrame(results, columns=columns)
    report.to_csv(FOLDER / "pid_comparison.csv", index=False)

    print(report.to_string(index=False))
    print("\nSaved: benchmark_results/pamap2/pid_comparison.csv")