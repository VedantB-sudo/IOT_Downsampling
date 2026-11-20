
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

DATA = Path("data/intel/data.txt.gz")
OUT = Path("benchmark_results/intel")
FEATURES = ["temperature", "humidity", "light", "voltage"]
SIZE, LIMIT = 1000, 1200
PATTERN = [450] * 4 + [60] * 8 + [450] * 10


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset missing: {DATA}")

    names = ["date", "time", "epoch", "moteid"] + FEATURES
    df = pd.read_csv(DATA, sep=r"\s+", names=names)
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], errors="coerce"
    )
    df["moteid"] = pd.to_numeric(df["moteid"], errors="coerce")
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = df[df["moteid"].between(1, 54) & df["timestamp"].notna()].copy()
    df["source_row"] = df.index
    return df.sort_values(["moteid", "timestamp", "source_row"])


def get_windows(df, feature, minimum=256, gap=300):
    windows = []
    for mote, group in df.groupby("moteid"):
        valid = group[feature].notna() & np.isfinite(group[feature])
        seconds = group["timestamp"].diff().dt.total_seconds()
        breaks = (~valid | seconds.gt(gap) | seconds.le(0)).cumsum()

        for _, part in group[valid].groupby(breaks[valid]):
            x = part[feature].to_numpy(float)
            rows = part["source_row"].to_numpy()

            for start in range(0, len(x), SIZE):
                signal = x[start:start + SIZE]
                if len(signal) >= minimum:
                    windows.append((int(mote), int(rows[start]), signal))
    return windows


class PID:
    def __init__(self):
        self.keep, self.integral, self.previous = 0.30, 0.0, 0.0

    def update(self, occupancy):
        error = 0.25 - occupancy
        derivative = error - self.previous
        candidate = np.clip(self.integral + error, -2, 2)
        trial = 0.30 + 0.55 * error + 0.05 * candidate + 0.10 * derivative

        if (0.05 <= trial <= 0.50 or
                (trial > 0.50 and error < 0) or
                (trial < 0.05 and error > 0)):
            self.integral = candidate

        target = 0.30 + 0.55 * error + 0.05 * self.integral + 0.10 * derivative
        self.keep = np.clip(
            self.keep + np.clip(target - self.keep, -0.08, 0.08), 0.05, 0.50
        )
        self.previous = error
        return self.keep


def make_schedule(n):
    rng = np.random.default_rng(42)
    capacity = (PATTERN * ((n + len(PATTERN) - 1) // len(PATTERN)))[:n]
    capacity = [max(10, int(c * rng.uniform(0.8, 1.2))) for c in capacity]

    pid, queue, counts = PID(), 0, []
    for available in capacity:
        count = round(SIZE * pid.update(queue / LIMIT))
        counts.append(count)
        queue = min(LIMIT, max(0, queue + count - available))

    base, rest = divmod(sum(counts), n)
    fixed = [
        base + (i + 1) * rest // n - i * rest // n
        for i in range(n)
    ]
    return capacity, counts, fixed


def network(counts, capacity):
    queue, drops, occupancy = 0, 0, []
    for count, available in zip(counts, capacity):
        queue = max(0, queue + count - available)
        drops += max(0, queue - LIMIT)
        queue = min(queue, LIMIT)
        occupancy.append(queue / LIMIT)
    return drops, 100 * np.mean(occupancy)


def transmit(samples, capacity):
    queue = deque()
    received = [set() for _ in samples]
    drops = 0

    for w, (indices, available) in enumerate(zip(samples, capacity)):
        queue.extend((w, int(i)) for i in indices)
        for _ in range(min(available, len(queue))):
            origin, index = queue.popleft()
            received[origin].add(index)
        while len(queue) > LIMIT:
            queue.pop()
            drops += 1

    for origin, index in queue:
        received[origin].add(index)
    return received, drops


def score(windows, selections):
    rmse, prdn, events = [], [], []
    for x, indices in zip(windows, selections):
        idx = np.array(sorted(indices), dtype=int)
        restored = np.interp(np.arange(len(x)), idx, x[idx]) if len(idx) else np.zeros_like(x)
        error = np.sqrt(np.mean((x - restored) ** 2))
        changes = np.abs(np.diff(x))
        important = np.where(changes > np.percentile(changes, 95))[0] + 1

        rmse.append(error)
        prdn.append(100 * error / max(x.std(), 1e-12))
        events.append(100 * np.isin(important, idx).mean() if len(important) else np.nan)

    return np.mean(rmse), np.mean(prdn), (
        np.nanmean(events) if np.isfinite(events).any() else np.nan
    )