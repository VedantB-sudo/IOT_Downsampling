
import pandas as pd
from pathlib import Path
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

folder = Path("benchmark_results/pamap2")

baseline = pd.read_csv(folder / "full_dataset/benchmark_summary.csv")
pid = pd.read_csv(folder / "pid_comparison.csv")
delivery = pd.read_csv(folder / "delivery_comparison.csv")


# Baseline results
base = baseline[[
    "sensor_type", "algorithm", "target_reduction_pct",
    "RMSE_mean", "PRDN_pct_mean", "P95_retained_pct_mean"
]].copy()

base.columns = [
    "Sensor", "Algorithm", "Reduction (%)",
    "RMSE", "PRDN (%)", "Event Retention (%)"
]
base.insert(3, "Retained (%)", 100 - base["Reduction (%)"])
base = base.sort_values(["Sensor", "Reduction (%)", "Algorithm"]).reset_index(drop=True)


# PID and Fixed results after network delivery
compare = pid.merge(
    delivery,
    on=["Sensor", "Algorithm", "Mode", "Drops"],
    validate="one_to_one"
)

compare = compare[[
    "Sensor", "Algorithm", "Mode", "Retained %",
    "PRDN After", "RMSE After", "Events After", "Drops"
]]

compare.columns = [
    "Sensor", "Algorithm", "Mode", "Retained (%)",
    "PRDN (%)", "RMSE", "Event Retention (%)", "Queue Drops"
]
compare = compare.sort_values(["Sensor", "Mode", "Algorithm"]).reset_index(drop=True)


# Pareto comparison: lower PRDN and higher event retention
def add_pareto(df, groups):
    df = df.copy()
    df["Good Trade-off?"] = "No"

    for _, group in df.groupby(groups):
        for i, row in group.iterrows():
            dominated = (
                (group["PRDN (%)"] <= row["PRDN (%)"]) &
                (group["Event Retention (%)"] >= row["Event Retention (%)"]) &
                (
                    (group["PRDN (%)"] < row["PRDN (%)"]) |
                    (group["Event Retention (%)"] > row["Event Retention (%)"])
                )
            ).any()

            if not dominated:
                df.loc[i, "Good Trade-off?"] = "Yes"

    return df


base = add_pareto(base, ["Sensor", "Reduction (%)"])
compare = add_pareto(compare, ["Sensor", "Mode"])


# Find winners within each group
def find_winners(df, groups):
    rows = []

    for key, group in df.groupby(groups):
        quality = group.loc[group["PRDN (%)"].idxmin()]
        events = group.loc[group["Event Retention (%)"].idxmax()]

        if not isinstance(key, tuple):
            key = (key,)

        rows.append([
            *key,
            quality["Algorithm"], quality["PRDN (%)"],
            events["Algorithm"], events["Event Retention (%)"]
        ])

    return pd.DataFrame(rows, columns=[
        *groups, "Best Reconstruction", "Lowest PRDN (%)",
        "Best Event Retention", "Highest Events (%)"
    ])


winners = find_winners(base, ["Sensor", "Reduction (%)"])
pid_winners = find_winners(compare, ["Sensor", "Mode"])


# Overall sensor winners across reduction levels
averages = base.groupby(
    ["Sensor", "Algorithm"], as_index=False
)[["PRDN (%)", "Event Retention (%)"]].mean()

overall = find_winners(averages, ["Sensor"])


# Excel sheets
sheets = {
    "Baseline Comparison": base,
    "Winners by Reduction": winners,
    "Overall Sensor Winners": overall,
    "PID vs Fixed": compare,
    "PID Fixed Winners": pid_winners
}

green = PatternFill("solid", fgColor="D9EAD3")
yellow = PatternFill("solid", fgColor="FFF2CC")
blue = PatternFill("solid", fgColor="17365D")
light_blue = PatternFill("solid", fgColor="DDEBF7")

output = Path("benchmark_results/Results_Analysis.xlsx")

with pd.ExcelWriter(output, engine="openpyxl") as writer:

    for name, df in sheets.items():
        df.round(2).to_excel(writer, sheet_name=name, index=False)
        ws = writer.sheets[name]

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for cell in ws[1]:
            cell.fill = blue
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center")

        for col in ws.columns:
            letter = get_column_letter(col[0].column)
            width = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[letter].width = min(width + 3, 27)

        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.00"

        # Colour winning results
        if name in ["Baseline Comparison", "PID vs Fixed"]:
            groups = (
                ["Sensor", "Reduction (%)"]
                if name == "Baseline Comparison"
                else ["Sensor", "Mode"]
            )

            headers = list(df.columns)
            prdn_col = headers.index("PRDN (%)") + 1
            event_col = headers.index("Event Retention (%)") + 1

            for _, group in df.groupby(groups):
                best_prdn = group["PRDN (%)"].min()
                best_events = group["Event Retention (%)"].max()

                for index, item in group.iterrows():
                    row = index + 2

                    if item["PRDN (%)"] == best_prdn:
                        ws.cell(row, prdn_col).fill = green
                        ws.cell(row, prdn_col).font = Font(bold=True)

                    if item["Event Retention (%)"] == best_events:
                        ws.cell(row, event_col).fill = yellow
                        ws.cell(row, event_col).font = Font(bold=True)

                    if item["Good Trade-off?"] == "Yes":
                        ws.cell(row, len(headers)).fill = light_blue

        if name in ["Winners by Reduction", "Overall Sensor Winners",
                    "PID Fixed Winners"]:

            headers = list(df.columns)

            for row in range(2, ws.max_row + 1):
                for heading, colour in [
                    ("Best Reconstruction", green),
                    ("Lowest PRDN (%)", green),
                    ("Best Event Retention", yellow),
                    ("Highest Events (%)", yellow)
                ]:
                    col = headers.index(heading) + 1
                    ws.cell(row, col).fill = colour
                    ws.cell(row, col).font = Font(bold=True)

        # Legend below each table
        last_row = ws.max_row
        row = last_row + 3

        ws.cell(row, 1, "LEGEND")
        ws.cell(row, 1).font = Font(
            bold=True, color="17365D", size=12
        )

        legend = [
            ("Green", "Best reconstruction: lowest PRDN (%)."),
            ("Yellow", "Best event retention: highest percentage of important changes."),
            ("Blue", "Good trade-off: no other algorithm performs better on both metrics."),
            ("PID", "Retention changes with previous network queue occupancy."),
            ("Fixed", "Retention stays constant at the matched average budget."),
            ("PRDN", "Normalised reconstruction error. Lower is better."),
            ("Event Retention", "P95 signal-change retention. Higher is better.")
        ]

        for label, meaning in legend:
            row += 1
            ws.cell(row, 1, label).font = Font(bold=True)
            ws.cell(row, 2, meaning)

            if label == "Green":
                ws.cell(row, 1).fill = green
            elif label == "Yellow":
                ws.cell(row, 1).fill = yellow
            elif label == "Blue":
                ws.cell(row, 1).fill = light_blue

        print(f"Added: {name} ({len(df)} rows)")

    # PRDN graphs for each sensor
    ws = writer.book.create_sheet("Graphs")
    ws["A1"] = "Signal Quality vs Data Reduction"
    ws["A1"].font = Font(bold=True, size=14, color="17365D")
    ws["A2"] = "Lower PRDN (%) means better reconstruction."

    start = 4

    for sensor, group in base.groupby("Sensor"):
        pivot = group.pivot_table(
            index="Reduction (%)",
            columns="Algorithm",
            values="PRDN (%)"
        ).sort_index()

        data_col = 12
        ws.cell(start, data_col, "Reduction (%)")

        for j, algorithm in enumerate(pivot.columns, data_col + 1):
            ws.cell(start, j, algorithm)

        for i, (reduction, values) in enumerate(
            pivot.iterrows(), start + 1
        ):
            ws.cell(i, data_col, reduction)

            for j, value in enumerate(values, data_col + 1):
                ws.cell(i, j, float(value))

        chart = LineChart()
        chart.title = f"{sensor}: PRDN vs Data Reduction"
        chart.y_axis.title = "PRDN (%) - Lower is better"
        chart.x_axis.title = "Data Reduction (%)"
        chart.width = 20
        chart.height = 10
        chart.style = 13

        data = Reference(
            ws, min_col=data_col + 1,
            max_col=data_col + len(pivot.columns),
            min_row=start, max_row=start + len(pivot)
        )
        categories = Reference(
            ws, min_col=data_col,
            min_row=start + 1,
            max_row=start + len(pivot)
        )

        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)

        colors = {
            "uniform": "2563EB",
            "std_dev": "EA580C",
            "hp_pe": "9333EA",
            "hp_energy": "DC2626",
            "hybrid": "16A34A",
            "spaced_hybrid": "0891B2"
        }

        for series, algorithm in zip(chart.series, pivot.columns):
            series.graphicalProperties.line.solidFill = colors.get(
                algorithm, "64748B"
            )
            series.graphicalProperties.line.width = 30000

        ws.add_chart(chart, f"A{start}")

        start += 22

    ws.column_dimensions["A"].width = 25
    print("Added: Graphs")

print(f"\nExcel report created: {output}")