
import numpy as np
import pandas as pd
from intel_common import OUT

path = OUT / "full_dataset/benchmark_summary.csv"
if not path.exists():
    raise FileNotFoundError("Run benchmark_intel.py first")

df = pd.read_csv(path)

metrics = {
    "Best PRDN": ("PRDN_pct_mean", "min"),
    "Best RMSE": ("RMSE_mean", "min"),
    "Best MAE": ("MAE_mean", "min"),
    "Best P95 Retention": ("P95_retained_pct_mean", "max"),
    "Fastest Algorithm": ("selection_time_ms_mean", "min")
}

rows = []

for (sensor, reduction), group in df.groupby(
    ["sensor_type", "target_reduction_pct"]
):
    row = {"Sensor": sensor, "Reduction %": reduction}

    for title, (column, direction) in metrics.items():
        valid = group.dropna(subset=[column])
        if valid.empty:
            row[title] = "N/A"
            continue

        best = valid[column].min() if direction == "min" else valid[column].max()
        winners = valid[np.isclose(valid[column], best)]["algorithm"]
        row[title] = ", ".join(winners)

    # Non-dominated PRDN and P95 combinations
    valid = group.dropna(
        subset=["PRDN_pct_mean", "P95_retained_pct_mean"]
    )
    pareto = []

    for _, candidate in valid.iterrows():
        dominated = (
            (valid["PRDN_pct_mean"] <= candidate["PRDN_pct_mean"])
            & (valid["P95_retained_pct_mean"] >= candidate["P95_retained_pct_mean"])
            & (
                (valid["PRDN_pct_mean"] < candidate["PRDN_pct_mean"])
                | (valid["P95_retained_pct_mean"] > candidate["P95_retained_pct_mean"])
            )
        ).any()

        if not dominated:
            pareto.append(candidate["algorithm"])

    row["Good Trade-off (Pareto)"] = ", ".join(pareto) or "N/A"
    rows.append(row)

winners = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
winners.to_csv(OUT / "algorithm_winners.csv", index=False)

with pd.ExcelWriter(OUT / "Intel_Results_Analysis.xlsx") as writer:
    df.to_excel(writer, sheet_name="Baseline Comparison", index=False)
    winners.to_excel(writer, sheet_name="Winners by Reduction", index=False)
    pd.read_csv(OUT / "pid_comparison.csv").to_excel(
        writer, sheet_name="PID vs Fixed", index=False
    )
    pd.read_csv(OUT / "delivery_comparison.csv").to_excel(
        writer, sheet_name="Delivery Comparison", index=False
    )

print(winners.to_string(index=False))
print("\nSaved:", OUT / "Intel_Results_Analysis.xlsx")