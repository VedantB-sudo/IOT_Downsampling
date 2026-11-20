
from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = Path("data/intel/data.txt.gz")

COLUMNS = [
    "date", "time", "epoch", "moteid",
    "temperature", "humidity", "light", "voltage"
]

FEATURES = ["temperature", "humidity", "light", "voltage"]

# Conservative physical bounds; adjust only with justification.
LIMITS = {
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "light": (0, 100000),
    "voltage": (0, 4)
}


def load_intel():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Intel dataset not found: {DATA_FILE}"
        )

    print("Loading Intel Berkeley Lab dataset...")

    df = pd.read_csv(
        DATA_FILE,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        compression="infer",
        low_memory=False
    )

    total_rows = len(df)

    for column in ["epoch", "moteid"] + FEATURES:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )

    df = df[
        df["moteid"].between(1, 54)
        & df["timestamp"].notna()
    ].copy()

    df = df.sort_values(["moteid", "timestamp"])

    print(f"Original rows: {total_rows:,}")
    print(f"Rows with valid mote IDs and times: {len(df):,}")

    return df


def get_windows(
    df,
    feature="temperature",
    window_size=1000,
    max_gap_seconds=155
):
    if feature not in FEATURES:
        raise ValueError(f"Choose from: {FEATURES}")

    low, high = LIMITS[feature]
    windows = []

    for mote_id, group in df.groupby("moteid"):
        group = group[
            group[feature].notna()
            & group[feature].between(low, high)
        ].copy()

        # Split sequences at long recording gaps.
        gaps = group["timestamp"].diff().dt.total_seconds()
        segments = gaps.gt(max_gap_seconds).cumsum()

        for _, segment in group.groupby(segments):
            values = segment[feature].to_numpy(dtype=float)

            # Keep complete, non-overlapping windows.
            for start in range(
                0, len(values) - window_size + 1,
                window_size
            ):
                window = values[start:start + window_size]
                if np.all(np.isfinite(window)):
                    windows.append(window)

    return windows


if __name__ == "__main__":
    df = load_intel()

    print("\nINTEL DATASET PREVIEW")
    print(df[COLUMNS].head(5).to_string(index=False))

    print("\nELIGIBLE WINDOWS")
    for feature in FEATURES:
        windows = get_windows(df, feature=feature)
        count = len(windows)

        print(
            f"{feature:<12} "
            f"{count:>6,} windows | "
            f"{count * 1000:>10,} samples"
        )