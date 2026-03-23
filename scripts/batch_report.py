#!/usr/bin/env python
"""
Generate a 2-page PDF report from batch-nutrients output.

Usage:
    python scripts/batch_report.py --run-dir output/run_20260317_114741
    python scripts/batch_report.py --run-dir output/run_20260317_114741 --output report.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *keywords: str) -> str | None:
    """Return first column whose name contains ALL keywords (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    for col in df.columns:
        c = col.lower()
        if all(k in c for k in kws):
            return col
    return None


# ---------------------------------------------------------------------------
# (Sankey page removed)

# ---------------------------------------------------------------------------
# Page 1 — Top 10 meals and top 10 items
# ---------------------------------------------------------------------------

def page1_top10(detail: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle("Top 10 Most Consumed Meals & Food Items", fontsize=14)

    # Top meals: count distinct (row, meal) pairs per meal text
    if "row" in detail.columns and "meal" in detail.columns:
        meal_freq = (
            detail[["row", "meal"]].drop_duplicates()
            .groupby("meal").size()
            .sort_values(ascending=False).head(10)
        )
    else:
        meal_freq = detail["meal"].value_counts().head(10)

    labels1 = [m[:55] + "…" if len(m) > 55 else m for m in meal_freq.index]
    ax1.barh(range(len(meal_freq)), meal_freq.values, color=plt.cm.Blues_r(np.linspace(0.3, 0.9, len(meal_freq))))
    ax1.set_yticks(range(len(meal_freq)))
    ax1.set_yticklabels(labels1, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_title("Top 10 Meals")
    ax1.set_xlabel("Times consumed")
    for i, v in enumerate(meal_freq.values):
        ax1.text(v + 0.1, i, str(v), va="center", fontsize=8)

    # Top items
    item_col = "ingredient" if "ingredient" in detail.columns else "usda_match"
    item_freq = detail[item_col].value_counts().head(10)
    labels2 = [i[:50] + "…" if len(i) > 50 else i for i in item_freq.index]
    ax2.barh(range(len(item_freq)), item_freq.values, color=plt.cm.Greens_r(np.linspace(0.3, 0.9, len(item_freq))))
    ax2.set_yticks(range(len(item_freq)))
    ax2.set_yticklabels(labels2, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_title("Top 10 Food Items")
    ax2.set_xlabel("Times consumed")
    for i, v in enumerate(item_freq.values):
        ax2.text(v + 0.1, i, str(v), va="center", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 2 — Nutrient distribution histograms
# ---------------------------------------------------------------------------

NUTRIENT_SPECS = [
    # (search keywords,            title,               subtitle,                                     group)
    (("energy", "kcal"),           "Energy (kcal)",     "Total energy per serving",                   "macro"),
    (("carbohydrate", "differ"),   "Carbohydrates (g)", "Total carbs, including complex carbs and sugars", "macro"),
    (("protein",),                 "Protein (g)",       "Total protein per serving",                  "macro"),
    (("sugars", "total"),          "Sugars (g)",        "Total sugars (free + added)",                "macro"),
    (("lipid", "fat"),             "Total Fat (g)",     "Total lipid content per serving",            "macro"),
    (("fiber", "dietary"),         "Dietary Fiber (g)", "Total dietary fibre per serving",            "macro"),
    (("iron", "fe"),               "Iron (mg)",         "Iron (Fe) content per serving",              "micro"),
    (("pantothenic",),             "Vitamin B5 (mg)",   "Pantothenic acid per serving",               "micro"),
    (("magnesium", "mg"),          "Magnesium (mg)",    "Magnesium (Mg) per serving",                 "micro"),
    (("vitamin b-12",),            "Vitamin B12 (µg)",  "Cobalamin per serving",                      "micro"),
]

BAR_COLOR = "#4589c6"   # matches screenshot blue


def page2_nutrients(detail: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    # Use meal-level totals: prefer summary CSV, else aggregate detail by meal
    if not summary.empty:
        meal_df = summary
    else:
        group_cols = [c for c in ("row", "meal") if c in detail.columns]
        num_cols = detail.select_dtypes(include="number").columns.tolist()
        meal_df = detail.groupby(group_cols)[num_cols].sum().reset_index()

    found = []
    for keywords, title, subtitle, group in NUTRIENT_SPECS:
        col = _find_col(meal_df, *keywords)
        if col:
            found.append((col, title, subtitle, group))

    if not found:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.5, "No matching nutrient columns found.", ha="center", va="center")
        return fig

    n = len(found)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.0 * nrows))
    fig.suptitle("Nutrient Distribution per Meal", fontsize=13, y=1.01)
    axes_flat = np.array(axes).flatten()

    for i, (col, title, subtitle, group) in enumerate(found):
        ax = axes_flat[i]
        data = meal_df[col].dropna()
        data = data[data > 0].values

        if len(data) < 3:
            ax.set_visible(False)
            continue

        # Clip extreme outliers at 99th percentile for readability
        p99 = np.percentile(data, 99)
        plot_data = data[data <= p99]

        ax.hist(plot_data, bins=20, color=BAR_COLOR, edgecolor="white", linewidth=0.4)

        # Clean minimal style matching the screenshot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7.5, length=3)
        ax.set_xlim(left=0)

        # Bold title + small grey subtitle above
        ax.set_title(title, fontsize=9, fontweight="bold", pad=16, loc="left")
        ax.text(0, 1.10, subtitle, transform=ax.transAxes,
                fontsize=6.5, color="#666", va="bottom")

        # MACRO / MICRO badge
        badge = "MACRO" if group == "macro" else "MICRO"
        badge_col = "#d0e8ff" if group == "macro" else "#ffd6ec"
        ax.text(0.98, 0.96, badge, transform=ax.transAxes,
                fontsize=5.5, color="#444", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=badge_col, edgecolor="none"))

        # Dashed median line + label
        med = np.median(data)
        if med <= p99:
            ax.axvline(med, color="#e05c2a", linewidth=1.3, linestyle="--", alpha=0.85)
            ylim = ax.get_ylim()
            ax.text(med, ylim[1] * 0.88, f" {med:.2g}",
                    fontsize=6, color="#e05c2a", va="top")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    from matplotlib.lines import Line2D
    fig.legend(
        handles=[Line2D([0], [0], color="#e05c2a", linewidth=1.3, linestyle="--")],
        labels=["Median"],
        loc="lower right", fontsize=8, frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(run_dir: Path, output_pdf: Path | None = None) -> None:
    detail_files  = sorted(run_dir.glob("*_detail.csv"))
    summary_files = sorted(run_dir.glob("*_summary.csv"))

    if not detail_files:
        print(f"[error] No *_detail.csv found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    detail  = pd.concat([pd.read_csv(f) for f in detail_files],  ignore_index=True)
    summary = (
        pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
        if summary_files else pd.DataFrame()
    )

    if output_pdf is None:
        output_pdf = run_dir / "report.pdf"

    n_meals = detail[["row", "meal"]].drop_duplicates()["row"].nunique() if "row" in detail.columns else "?"
    print(f"Loaded {len(detail)} item rows across {n_meals} meals")
    print(f"Writing → {output_pdf}\n")

    with pdf_backend.PdfPages(output_pdf) as pdf:
        print("  Page 1: Top 10 meals & items…")
        fig1 = page1_top10(detail)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        print("  Page 2: Nutrient distributions (per meal)…")
        fig2 = page2_nutrients(detail, summary)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"\nDone → {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate batch analysis PDF report")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to run folder containing *_detail.csv and *_summary.csv",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PDF path (default: <run-dir>/report.pdf)",
    )
    args = parser.parse_args()
    generate_report(Path(args.run_dir), Path(args.output) if args.output else None)
