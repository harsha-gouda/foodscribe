"""
Standalone ingredient nutrient lookup — no LLM step.

Reads a CSV with ingredient descriptions and gram weights, runs semantic
retrieval, and writes a nutrient profile CSV scaled to each ingredient's gram weight.

Usage:
    python scripts/ingredient_lookup.py input/food_ingredients_GutPuzzle.csv

    # Custom column names or output path
    python scripts/ingredient_lookup.py input/foods.csv \
        --ingredient-col FoodName --grams-col WeightG \
        --output output/foods_nutrients.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("FOODSCRIBE_DATA_DIR", "data/"))


def _s(v) -> str:
    if v is None:
        return ""
    try:
        import math
        if math.isnan(float(v)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingredient-level USDA nutrient lookup")
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("--ingredient-col", default="Ingredient", help="Column with ingredient text (default: Ingredient)")
    parser.add_argument("--grams-col", default="grams", help="Column with gram weights (default: grams)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <stem>_nutrients.csv)")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieval candidates per ingredient (default: 3)")
    parser.add_argument("--data-dir", default=None, help="Path to data/ folder")
    parser.add_argument("--overrides", default=None,
                        help="CSV with columns 'ingredient,fdc_id' for manual FDC overrides "
                             "(default: input/fdc_overrides.csv if it exists)")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    ddir = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Load FDC overrides: ingredient text (exact, stripped) -> fdc_id
    overrides_path = Path(args.overrides) if args.overrides else in_path.parent / "fdc_overrides.csv"
    fdc_overrides: dict[str, int] = {}
    if overrides_path.exists():
        ov_df = pd.read_csv(overrides_path)
        fdc_overrides = {
            str(row["ingredient"]).strip().lower(): int(row["fdc_id"])
            for _, row in ov_df.iterrows()
        }
        print(f"Loaded {len(fdc_overrides)} FDC override(s) from {overrides_path}")

    df = pd.read_csv(in_path)
    for col in (args.ingredient_col, args.grams_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    # Extra columns to carry through
    pipeline_cols = {args.ingredient_col, args.grams_col}
    extra_cols = [c for c in df.columns if c not in pipeline_cols]

    # Load retriever and nutrient lookup
    openai_index = ddir / "food_embeddings_openai.npy"
    if openai_index.exists():
        from foodscribe.retrieval.openai_retriever import OpenAIRetriever
        retriever = OpenAIRetriever(data_dir=ddir, top_k=args.top_k)
    else:
        from foodscribe.retrieval.mpnet_retriever import MPNetRetriever
        retriever = MPNetRetriever(data_dir=ddir, top_k=args.top_k)

    from foodscribe.nutrients.categories import CategoryLookup
    from foodscribe.nutrients.lookup import NutrientLookup

    cat_lookup = CategoryLookup(data_dir=ddir)
    nut_lookup = NutrientLookup(data_dir=ddir, category_lookup=cat_lookup)

    ingredients = df[args.ingredient_col].tolist()
    grams_list = df[args.grams_col].tolist()

    print(f"Retrieving USDA matches for {len(ingredients)} ingredients...")
    batch_results = retriever.retrieve_batch(ingredients, top_k=args.top_k)

    detail_records = []
    for i, (ingredient, grams, cands) in enumerate(zip(ingredients, grams_list, batch_results)):
        extra = {c: df.iloc[i][c] for c in extra_cols}
        if not cands:
            print(f"  [no match] {ingredient}")
            continue

        # Apply manual FDC override if one exists for this ingredient
        override_fdc = fdc_overrides.get(ingredient.strip().lower())
        if override_fdc:
            fdc_id = override_fdc
            match_desc = f"[override] fdc={override_fdc}"
            match_score = 1.0
        else:
            fdc_id = cands[0].fdc_id
            match_desc = cands[0].description
            match_score = cands[0].score

        grams_val = float(grams) if pd.notna(grams) else None
        row = nut_lookup.get_scaled(fdc_id, grams_val) if grams_val is not None else nut_lookup.get(fdc_id)
        # Populate match_desc from nutrient lookup if we used an override
        if override_fdc and row:
            match_desc = row.description
        if not row:
            print(f"  [no nutrients] {ingredient} -> {match_desc}")
            continue
        nutrient_cols = {k: round(v, 4) for k, v in (row.all_nutrients or {}).items()}
        detail_records.append({
            **extra,
            args.ingredient_col: ingredient,
            args.grams_col: grams_val or 100,
            "fdc_id":      fdc_id,
            "usda_match":  match_desc,
            "match_score": round(match_score, 4),
            "category":    _s(getattr(row, "category", None)),
            "subcategory": _s(getattr(row, "subcategory", None)),
            "data_type":   _s(getattr(row, "data_type", None)),
            **nutrient_cols,
        })

    if not detail_records:
        print("[error] No nutrients retrieved — check your input column names and data directory.")
        return

    out_df = pd.DataFrame(detail_records)
    out_path = Path(args.output) if args.output else in_path.with_stem(in_path.stem + "_nutrients")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nMatched {len(detail_records)}/{len(ingredients)} ingredients")
    print(f"Output -> {out_path}")


if __name__ == "__main__":
    main()
