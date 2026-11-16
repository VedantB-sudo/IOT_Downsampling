
import pandas as pd
from pathlib import Path
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

folder = Path("benchmark_results")
files = {
    "Baseline Results": folder / "pamap2/full_dataset/benchmark_summary.csv",
    "PID Comparison": folder / "pamap2/pid_comparison.csv",
    "Network Delivery": folder / "pamap2/delivery_comparison.csv"
}

columns = {
    "Baseline Results": {
        "Algorithm": "algorithm",
        "Sensor / Signal": "sensor_type",
        "Data Reduction (%)": "target_reduction_pct",
        "PRDN (%)": "PRDN_pct_mean",
        "RMSE": "RMSE_mean",
        "MAE": "MAE_mean",
        "Event Retention (%)": "P95_retained_pct_mean",
        "Processing Time (ms)": "selection_time_ms_mean",
        "Windows Tested": "window_count"
    },
    "PID Comparison": {
        "Sensor / Signal": "Sensor",
        "Algorithm": "Algorithm",
        "Mode": "Mode",
        "Average Retained (%)": "Retained %",
        "RMSE": "RMSE",
        "PRDN (%)": "PRDN %",
        "Event Retention (%)": "P95 event %",
        "Queue Drops": "Drops",
        "Mean Queue (%)": "Mean queue %"
    },
    "Network Delivery": {
        "Sensor / Signal": "Sensor",
        "Algorithm": "Algorithm",
        "Mode": "Mode",
        "RMSE Before": "RMSE Before",
        "RMSE After": "RMSE After",
        "PRDN Before (%)": "PRDN Before",
        "PRDN After (%)": "PRDN After",
        "Events Before (%)": "Events Before",
        "Events After (%)": "Events After",
        "Queue Drops": "Drops"
    }
}

metrics = {
    "Data Retained (%)": "Percentage of original readings kept.",
    "Data Reduction (%)": "Percentage of original readings removed.",
    "Average Retained (%)": "Average percentage kept across all windows.",
    "RMSE": "Root Mean Square Error. Lower is better.",
    "MAE": "Mean Absolute Error. Lower is better.",
    "PRDN (%)": "Normalised reconstruction error. Lower is better.",
    "Event Retention (%)": "Percentage of large signal changes retained.",
    "Processing Time (ms)": "Average time to select readings per window.",
    "Windows Tested": "Number of windows evaluated.",
    "Mean Queue (%)": "Average network queue occupancy.",
    "Queue Drops": "Readings lost when the simulated queue becomes full.",
    "Mode": "PID adapts retention; Fixed uses a constant retention rate."
}

algorithms = {
    "uniform": "Uniform Decimation: selects readings at regular intervals.",
    "std_dev": "Standard Deviation: selects readings based on signal variation.",
    "hp_pe": "High-Pass Permutation Entropy: uses changes in signal complexity.",
    "hp_energy": "High-Pass Energy: uses high-frequency signal energy.",
    "hybrid": "Combines HP-PE and HP-Energy for sample selection.",
    "spaced_hybrid": "Hybrid selection with a spacing constraint to distribute selected readings."
}

output = folder / "Dissertation_Results.xlsx"

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    for name, path in files.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue

        df = pd.read_csv(path)
        missing = [c for c in columns[name].values() if c not in df.columns]

        if missing:
            print(f"{name}: missing columns {missing}")
            print("Run the updated multi-sensor benchmarks first.")
            continue

        result = df[list(columns[name].values())].copy()
        result.columns = list(columns[name])

        if name == "Baseline Results":
            result.insert(2, "Data Retained (%)",
                          100 - result["Data Reduction (%)"])

        result = result.round(2)
        result.to_excel(writer, sheet_name=name, index=False)

        ws = writer.sheets[name]
        ws.freeze_panes = "A2"
        last_row = ws.max_row
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        for col in ws.columns:
            letter = get_column_letter(col[0].column)
            width = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[letter].width = min(width + 3, 28)

        # Metric explanations
        row = last_row + 3
        ws.cell(row, 1, "METRIC EXPLANATIONS")
        ws.cell(row, 1).font = Font(bold=True, color="17365D", size=12)

        for heading in result.columns:
            meaning = metrics.get(heading)

            if meaning is None and ("Before" in heading or "After" in heading):
                meaning = (
                    "Metric measured before simulated network losses."
                    if "Before" in heading else
                    "Metric measured after simulated network losses."
                )

            if meaning:
                row += 1
                ws.cell(row, 1, heading).font = Font(bold=True)
                ws.cell(row, 2, meaning)

        # Algorithm explanations
        row += 3
        ws.cell(row, 1, "ALGORITHM EXPLANATIONS")
        ws.cell(row, 1).font = Font(bold=True, color="17365D", size=12)

        for algorithm, meaning in algorithms.items():
            row += 1
            ws.cell(row, 1, algorithm).font = Font(bold=True)
            ws.cell(row, 2, meaning)

        ws.column_dimensions["B"].width = max(
            ws.column_dimensions["B"].width or 15, 32
        )

        for r in range(last_row + 3, row + 1):
            if ws.cell(r, 2).value:
                ws.merge_cells(start_row=r, start_column=2,
                               end_row=r, end_column=ws.max_column)
                ws.cell(r, 2).alignment = Alignment(
                    wrap_text=True, vertical="center"
                )
                ws.row_dimensions[r].height = 30

        print(f"Added: {name} ({len(result)} rows)")

print(f"\nExcel report created: {output}")