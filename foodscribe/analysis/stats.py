"""Meal statistical analysis and visualisation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from foodscribe.nutrients.lookup import NutrientRow


@dataclass
class MealSummary:
    total_energy_kcal: float
    total_protein_g: float
    total_carb_g: float
    total_fat_g: float
    total_fiber_g: float
    pct_protein_kcal: float
    pct_carb_kcal: float
    pct_fat_kcal: float
    item_count: int
    categories_present: list[str]
    dominant_category: str | None


def _f(v: float | None) -> float:
    return v if v is not None else 0.0


def _s(v) -> str:
    """Return v as a string, treating None and NaN as empty string."""
    if v is None:
        return ""
    try:
        import math
        if math.isnan(v):
            return ""
    except TypeError:
        pass
    return str(v)


class MealAnalyser:
    """Takes NutrientRow objects and computes summary stats and visualisations."""

    def summarise(self, rows: list[NutrientRow]) -> MealSummary:
        total_e = sum(_f(r.energy_kcal) for r in rows)
        total_p = sum(_f(r.protein_g) for r in rows)
        total_c = sum(_f(r.carb_g) for r in rows)
        total_f = sum(_f(r.fat_g) for r in rows)
        total_fi = sum(_f(r.fiber_g) for r in rows)

        macro_kcal = total_p * 4 + total_c * 4 + total_f * 9
        if macro_kcal > 0:
            pct_p = total_p * 4 / macro_kcal * 100
            pct_c = total_c * 4 / macro_kcal * 100
            pct_f = total_f * 9 / macro_kcal * 100
        else:
            pct_p = pct_c = pct_f = 0.0

        cats = sorted({r.category for r in rows if r.category and r.category != "Unknown"})

        cat_energy: dict[str, float] = defaultdict(float)
        for r in rows:
            if r.category:
                cat_energy[r.category] += _f(r.energy_kcal)
        dominant = max(cat_energy, key=cat_energy.get) if cat_energy else None

        return MealSummary(
            total_energy_kcal=total_e,
            total_protein_g=total_p,
            total_carb_g=total_c,
            total_fat_g=total_f,
            total_fiber_g=total_fi,
            pct_protein_kcal=round(pct_p, 1),
            pct_carb_kcal=round(pct_c, 1),
            pct_fat_kcal=round(pct_f, 1),
            item_count=len(rows),
            categories_present=cats,
            dominant_category=dominant,
        )

    def plot_category_breakdown(self, rows: list[NutrientRow], save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        cats = sorted({r.category or "Unknown" for r in rows})
        colors = {c: cm.tab10(i / max(len(cats), 1)) for i, c in enumerate(cats)}
        all_same = len(cats) == 1

        fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.7)))
        labels = [r.description[:40] or f"fdc_id {r.fdc_id}" for r in rows]
        for i, r in enumerate(rows):
            ax.barh(i, _f(r.energy_kcal), color=colors.get(r.category or "Unknown"))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Energy (kcal)")
        ax.set_title("Energy by Food Item (coloured by USDA category)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in cats]
        ax.legend(handles, cats, loc="lower right", fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_macros_pie(self, summary: MealSummary, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        sizes = [summary.pct_protein_kcal, summary.pct_carb_kcal, summary.pct_fat_kcal]
        labels = [
            f"Protein\n{summary.pct_protein_kcal:.1f}%",
            f"Carbs\n{summary.pct_carb_kcal:.1f}%",
            f"Fat\n{summary.pct_fat_kcal:.1f}%",
        ]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(sizes, labels=labels, colors=["#4e9af1", "#f4a261", "#e76f51"],
               autopct="%1.0f%%", startangle=90)
        ax.set_title(f"Macronutrient Split ({summary.total_energy_kcal:.0f} kcal total)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_nutrient_bars(self, rows: list[NutrientRow], save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt
        import numpy as np

        fields = ["energy_kcal", "protein_g", "carb_g", "fat_g"]
        data = np.array([[_f(getattr(r, n)) for n in fields] for r in rows], dtype=float)
        mx = data.max(axis=0)
        mx[mx == 0] = 1
        norm = data / mx
        x = np.arange(len(rows))
        w = 0.2
        fig, ax1 = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
        ax2 = ax1.twinx()
        pairs = [("Protein (g)", "#2a9d8f"), ("Carbs (g)", "#f4a261"), ("Fat (g)", "#e76f51")]
        for i, (lbl, col) in enumerate(pairs):
            ax1.bar(x + (i - 0.5) * w, norm[:, i + 1], w, label=lbl, color=col)
        ax2.bar(x - w, data[:, 0], w, label="Energy (kcal)", color="#4e9af1", alpha=0.6)
        fl = [r.description[:20] or f"fdc {r.fdc_id}" for r in rows]
        ax1.set_xticks(x)
        ax1.set_xticklabels(fl, rotation=30, ha="right")
        ax1.set_ylabel("Normalised (0-1)")
        ax2.set_ylabel("Energy (kcal)")
        ax1.set_title("Nutrient Profile per Food Item")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_energy_distribution(self, rows: list[NutrientRow], save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        total_e = sum(_f(r.energy_kcal) for r in rows) or 1
        energies = [_f(r.energy_kcal) for r in rows]
        labels = [r.description[:35] or f"fdc {r.fdc_id}" for r in rows]
        fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.7)))
        bars = ax.barh(range(len(rows)), energies, color="#4e9af1")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Energy (kcal)")
        ax.set_title("Energy Contribution per Food Item")
        for bar, e in zip(bars, energies):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{e:.0f} kcal ({e / total_e * 100:.1f}%)", va="center", fontsize=8)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def print_table(
        self,
        rows: list[NutrientRow],
        summary: MealSummary,
        show_category: bool = False,
        show_all_nutrients: bool = False,
    ) -> None:
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()
            t = Table(title="Meal Nutrient Profile", show_footer=True)
            t.add_column("Food (USDA match)", footer="TOTAL", style="bold")
            if show_category:
                t.add_column("Category", footer="")
                t.add_column("Subcategory", footer="")
            t.add_column("Grams", footer="", justify="right")
            t.add_column("Energy(kcal)", footer=f"{summary.total_energy_kcal:.0f}", justify="right")
            t.add_column("Protein(g)", footer=f"{summary.total_protein_g:.1f}", justify="right")
            t.add_column("Carb(g)", footer=f"{summary.total_carb_g:.1f}", justify="right")
            t.add_column("Fat(g)", footer=f"{summary.total_fat_g:.1f}", justify="right")
            t.add_column("Fiber(g)", footer=f"{summary.total_fiber_g:.1f}", justify="right")
            for r in rows:
                style = "yellow" if getattr(r, "_confidence", 5) <= 2 else ""
                grams_str = f"{r._grams:.0f}" if hasattr(r, "_grams") and r._grams else "100*"
                row_data = [r.description[:40]]
                if show_category:
                    row_data += [_s(r.category), _s(r.subcategory)]
                row_data += [
                    grams_str,
                    f"{_f(r.energy_kcal):.0f}",
                    f"{_f(r.protein_g):.1f}",
                    f"{_f(r.carb_g):.1f}",
                    f"{_f(r.fat_g):.1f}",
                    f"{_f(r.fiber_g):.1f}",
                ]
                t.add_row(*row_data, style=style)
            console.print(t)
            console.print(
                f"\nMacronutrient split:  Protein [bold]{summary.pct_protein_kcal:.0f}%[/bold]"
                f"  |  Carbs [bold]{summary.pct_carb_kcal:.0f}%[/bold]"
                f"  |  Fat [bold]{summary.pct_fat_kcal:.0f}%[/bold]"
            )
            if summary.categories_present:
                console.print(f"Categories present:   {chr(183).join(summary.categories_present)}")
            if summary.dominant_category:
                dom_e = sum(_f(r.energy_kcal) for r in rows if r.category == summary.dominant_category)
                dom_pct = dom_e / (summary.total_energy_kcal or 1) * 100
                console.print(
                    f"Dominant category:    {summary.dominant_category}"
                    f"  ({dom_e:.0f} kcal, {dom_pct:.0f}% of meal energy)"
                )

            if show_all_nutrients:
                # Aggregate all_nutrients across rows
                totals: dict[str, float] = {}
                for r in rows:
                    for nutrient, val in (r.all_nutrients or {}).items():
                        totals[nutrient] = totals.get(nutrient, 0.0) + val

                if totals:
                    nt = Table(title="Full Nutrient Panel (meal totals)", show_header=True)
                    nt.add_column("Nutrient", style="bold")
                    nt.add_column("Amount", justify="right")
                    for nutrient, val in sorted(totals.items()):
                        nt.add_row(nutrient, f"{val:.3g}")
                    console.print(nt)

        except ImportError:
            headers = ["Food", "kcal", "Prot(g)", "Carb(g)", "Fat(g)", "Fiber(g)"]
            if show_category:
                headers.insert(1, "Category")
            print("\t".join(headers))
            for r in rows:
                cols = [r.description[:35], f"{_f(r.energy_kcal):.0f}", f"{_f(r.protein_g):.1f}",
                        f"{_f(r.carb_g):.1f}", f"{_f(r.fat_g):.1f}", f"{_f(r.fiber_g):.1f}"]
                if show_category:
                    cols.insert(1, r.category or "")
                print("\t".join(cols))
            print(f"TOTAL\t{summary.total_energy_kcal:.0f}\t{summary.total_protein_g:.1f}"
                  f"\t{summary.total_carb_g:.1f}\t{summary.total_fat_g:.1f}\t{summary.total_fiber_g:.1f}")
