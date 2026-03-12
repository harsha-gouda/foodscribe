"""USDA food category lookup."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FoodCategory:
    fdc_id: int
    description: str
    category: str
    subcategory: str | None
    data_type: str  # "foundation" | "sr_legacy" | "survey_fndds"


class CategoryLookup:
    """
    Returns USDA food category metadata for a given fdc_id.

    Loading priority:
    1. data/food_categories.csv  (preferred — built by scripts/build_data.py)
    2. data/food_metadata.csv    (fallback — derives category from food_category col)
    3. All categories marked "Unknown" with a warning
    """

    def __init__(self, data_dir: str | Path = "data/") -> None:
        self.data_dir = Path(data_dir)
        self._index: dict[int, FoodCategory] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API (stubs — filled in Stage 4)
    # ------------------------------------------------------------------

    def get(self, fdc_id: int) -> FoodCategory | None:
        return self._index.get(fdc_id)

    def get_batch(self, fdc_ids: list[int]) -> dict[int, FoodCategory | None]:
        return {fdc_id: self._index.get(fdc_id) for fdc_id in fdc_ids}

    def list_categories(self) -> list[str]:
        return sorted({fc.category for fc in self._index.values()})

    def filter_by_category(self, category: str) -> list[FoodCategory]:
        needle = category.lower()
        return [fc for fc in self._index.values() if needle in fc.category.lower()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        cats_path = self.data_dir / "food_categories.csv"
        meta_path = self.data_dir / "food_metadata.csv"

        if cats_path.exists():
            df = pd.read_csv(cats_path, dtype={"fdc_id": int})
            self._build_index(df)
        elif meta_path.exists():
            logger.warning(
                "food_categories.csv not found; deriving categories from food_metadata.csv. "
                "Run scripts/build_data.py for the full categories file."
            )
            df = pd.read_csv(meta_path, dtype={"fdc_id": int})
            self._build_index_from_metadata(df)
        else:
            logger.warning(
                "Neither food_categories.csv nor food_metadata.csv found. "
                "All categories will be 'Unknown'."
            )

    def _build_index(self, df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            self._index[int(row.fdc_id)] = FoodCategory(
                fdc_id=int(row.fdc_id),
                description=getattr(row, "description", ""),
                category=getattr(row, "category", "Unknown") or "Unknown",
                subcategory=getattr(row, "subcategory", None) or None,
                data_type=getattr(row, "data_type", "unknown") or "unknown",
            )

    def _build_index_from_metadata(self, df: pd.DataFrame) -> None:
        for row in df.itertuples(index=False):
            cat = getattr(row, "food_category", None) or "Unknown"
            self._index[int(row.fdc_id)] = FoodCategory(
                fdc_id=int(row.fdc_id),
                description=getattr(row, "description", ""),
                category=cat,
                subcategory=None,
                data_type="unknown",
            )
