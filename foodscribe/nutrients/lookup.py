"""USDA nutrient value lookup from foods_wide.csv."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class NutrientRow:
    fdc_id: int
    description: str
    category: str | None
    subcategory: str | None
    data_type: str | None
    energy_kcal: float | None
    protein_g: float | None
    carb_g: float | None
    fat_g: float | None
    fiber_g: float | None
    sugar_g: float | None
    sodium_mg: float | None
    all_nutrients: dict[str, float] = field(default_factory=dict)


class NutrientLookup:
    """
    Reads foods_wide.csv (fdc_id as index, nutrient columns as float).
    Returns a NutrientRow for a given fdc_id.
    """

    # Maps logical field → actual column name in foods_wide.csv
    # Column names are "{nutrient_name} ({unit_name})" as built by build_data.py
    _NUTRIENT_COLS = {
        "energy_kcal":  "Energy (kcal)",
        "protein_g":    "Protein (g)",
        "carb_g":       "Carbohydrate, by difference (g)",
        "fat_g":        "Total lipid (fat) (g)",
        "fiber_g":      "Fiber, total dietary (g)",
        "sugar_g":      "Sugars, Total (g)",
        "sodium_mg":    "Sodium, Na (mg)",
    }

    # Kept for external reference / compatibility
    CORE_NUTRIENTS = list(_NUTRIENT_COLS.values())

    def __init__(
        self,
        data_dir: str | Path = "data/",
        category_lookup=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.category_lookup = category_lookup
        self._df: pd.DataFrame | None = None
        self._meta: pd.DataFrame | None = None
        self._load()

    # ------------------------------------------------------------------
    # Public API (stubs — filled in Stage 4)
    # ------------------------------------------------------------------

    def get(self, fdc_id: int) -> NutrientRow | None:
        if self._df is None or fdc_id not in self._df.index:
            return None
        return self._build_row(fdc_id, scale=1.0)

    def get_scaled(self, fdc_id: int, grams: float) -> NutrientRow | None:
        """USDA values are per 100 g; scale by grams/100."""
        if self._df is None or fdc_id not in self._df.index:
            return None
        return self._build_row(fdc_id, scale=grams / 100.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_row(self, fdc_id: int, scale: float) -> NutrientRow:
        row = self._df.loc[fdc_id]
        description = ""
        if self._meta is not None and fdc_id in self._meta.index:
            description = str(self._meta.loc[fdc_id, "description"])

        cat_obj = None
        if self.category_lookup is not None:
            cat_obj = self.category_lookup.get(fdc_id)

        def _val(col: str) -> float | None:
            if col not in self._df.columns:
                return None
            v = row.get(col)
            return None if (v is None or (isinstance(v, float) and v != v)) else float(v) * scale

        all_nutrients = {
            col: float(row[col]) * scale
            for col in self._df.columns
            if col != "fdc_id" and row.get(col) == row.get(col)  # exclude NaN
        }

        return NutrientRow(
            fdc_id=fdc_id,
            description=description,
            category=cat_obj.category if cat_obj else None,
            subcategory=cat_obj.subcategory if cat_obj else None,
            data_type=cat_obj.data_type if cat_obj else None,
            energy_kcal=_val(self._NUTRIENT_COLS["energy_kcal"]),
            protein_g=_val(self._NUTRIENT_COLS["protein_g"]),
            carb_g=_val(self._NUTRIENT_COLS["carb_g"]),
            fat_g=_val(self._NUTRIENT_COLS["fat_g"]),
            fiber_g=_val(self._NUTRIENT_COLS["fiber_g"]),
            sugar_g=_val(self._NUTRIENT_COLS["sugar_g"]),
            sodium_mg=_val(self._NUTRIENT_COLS["sodium_mg"]),
            all_nutrients=all_nutrients,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        wide_path = self.data_dir / "foods_wide.csv"
        meta_path = self.data_dir / "food_metadata.csv"
        if wide_path.exists():
            self._df = pd.read_csv(wide_path, index_col="fdc_id")
        if meta_path.exists():
            self._meta = pd.read_csv(meta_path, index_col="fdc_id")
