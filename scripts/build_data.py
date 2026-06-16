"""
One-off script to build all data/ files required by FoodScribe from raw USDA long-format CSVs
and optionally NHANES dietary intake data.

Input files (--usda-dir):
    foundation_long.csv   — USDA Foundation foods, long format
    legacy_long.csv       — USDA SR Legacy foods, long format
    survey_long.csv       — USDA Survey/FNDDS foods, long format

    All three share schema:
        fdc_id, data_type, ndb_number, description, food_category,
        nutrient_name, amount, unit_name, ...

Optional input (--nhanes-dir):
    DR1IFF_J_with_descriptions.csv  — NHANES 2021-2023 dietary intake records
    Unique foods not already in USDA are extracted, nutrients normalised to
    per-100g, and appended to all output tables.

Output files (--data-dir):
    food_metadata.csv          fdc_id, description, food_category, data_type
    food_categories.csv        fdc_id, description, category, subcategory, data_type
    foods_wide.csv             fdc_id + one column per nutrient (per 100 g)
    food_embeddings_mpnet.npy  (N, 768) float32, L2-normalised

Usage:
    python scripts/build_data.py --usda-dir ../USDA_data/ --data-dir data/
    python scripts/build_data.py --usda-dir ../USDA_data/ --nhanes-dir ../USDA_data/NHANES/ --data-dir data/
    python scripts/build_data.py --usda-dir ../USDA_data/ --nhanes-dir ../USDA_data/NHANES/ --embedder openai
    python scripts/build_data.py --usda-dir ../USDA_data/ --data-dir data/ --skip-embeddings
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_data")


# ---------------------------------------------------------------------------
# Data-type normalisation
# ---------------------------------------------------------------------------

_DATA_TYPE_MAP = {
    "foundation": "foundation",
    "sr legacy": "sr_legacy",
    "sr_legacy": "sr_legacy",
    "survey": "survey_fndds",
    "survey_fndds": "survey_fndds",
    "fndds": "survey_fndds",
}


def _normalise_data_type(raw: str) -> str:
    return _DATA_TYPE_MAP.get(str(raw).strip().lower(), "unknown")


# ---------------------------------------------------------------------------
# Step A — food_metadata.csv
# ---------------------------------------------------------------------------

def build_metadata(usda_dir: Path, data_dir: Path) -> pd.DataFrame:
    """
    Concatenate the three long CSVs, deduplicate by fdc_id, and write
    data/food_metadata.csv with columns: fdc_id, description, food_category, data_type.
    """
    log.info("Step A — building food_metadata.csv ...")
    frames = []
    for fname in ("foundation_long.csv", "legacy_long.csv", "survey_long.csv"):
        fpath = usda_dir / fname
        if not fpath.exists():
            log.warning("  %s not found — skipping", fpath)
            continue
        log.info("  reading %s ...", fname)
        df = pd.read_csv(
            fpath,
            usecols=lambda c: c in {
                "fdc_id", "data_type", "description", "food_category"
            },
            dtype=str,
            low_memory=False,
        )
        frames.append(df)

    if not frames:
        log.error("No USDA CSV files found in %s", usda_dir)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate: keep one row per fdc_id (first occurrence)
    meta = (
        combined[["fdc_id", "description", "food_category", "data_type"]]
        .drop_duplicates(subset="fdc_id")
        .copy()
    )
    meta["fdc_id"] = meta["fdc_id"].astype(int)
    meta["data_type"] = meta["data_type"].apply(_normalise_data_type)
    meta["food_category"] = meta["food_category"].fillna("Unknown")
    meta.sort_values("fdc_id", inplace=True)
    meta.reset_index(drop=True, inplace=True)

    out_path = data_dir / "food_metadata.csv"
    meta.to_csv(out_path, index=False)
    log.info("  wrote %s  (%d foods)", out_path, len(meta))
    return meta


# ---------------------------------------------------------------------------
# Step B — food_categories.csv
# ---------------------------------------------------------------------------

def build_categories(meta: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Build data/food_categories.csv from metadata.
    subcategory is inferred from the first comma-segment of the description.
    """
    log.info("Step B — building food_categories.csv ...")

    cats = meta[["fdc_id", "description", "food_category", "data_type"]].copy()
    cats.rename(columns={"food_category": "category"}, inplace=True)

    # Subcategory: text before the first comma in description (if a comma exists)
    cats["subcategory"] = cats["description"].apply(
        lambda d: d.split(",")[0].strip() if isinstance(d, str) and "," in d else None
    )

    # Reorder columns
    cats = cats[["fdc_id", "description", "category", "subcategory", "data_type"]]

    out_path = data_dir / "food_categories.csv"
    cats.to_csv(out_path, index=False)
    log.info("  wrote %s  (%d rows)", out_path, len(cats))
    return cats


# ---------------------------------------------------------------------------
# Step C — foods_wide.csv
# ---------------------------------------------------------------------------

def build_foods_wide(usda_dir: Path, data_dir: Path, fdc_ids: set[int]) -> pd.DataFrame:
    """
    Pivot long → wide: rows = fdc_id, columns = nutrient names.
    Reads all three long CSVs to get nutrient values.
    Files are read in priority order (Foundation > SR Legacy > Survey).
    For any (fdc_id, nutrient_name) duplicate across files, the first
    (highest-priority) value is kept.
    """
    log.info("Step C — building foods_wide.csv ...")
    frames = []
    # Priority order: Foundation > SR Legacy > Survey
    for fname in ("foundation_long.csv", "legacy_long.csv", "survey_long.csv"):
        fpath = usda_dir / fname
        if not fpath.exists():
            continue
        log.info("  reading %s (nutrients only) ...", fname)
        df = pd.read_csv(
            fpath,
            usecols=lambda c: c in {"fdc_id", "nutrient_name", "unit_name", "amount"},
            dtype={"fdc_id": str, "nutrient_name": str, "unit_name": str, "amount": str},
            low_memory=False,
        )
        frames.append(df)

    long_df = pd.concat(frames, ignore_index=True)
    long_df["fdc_id"] = long_df["fdc_id"].astype(int)
    long_df["amount"] = pd.to_numeric(long_df["amount"], errors="coerce")

    # Filter to only foods we have in metadata
    long_df = long_df[long_df["fdc_id"].isin(fdc_ids)]

    # Drop rows with null nutrient_name or amount
    long_df.dropna(subset=["nutrient_name", "amount"], inplace=True)

    # Normalise energy nutrient names.
    # Foundation data uses variants like "Energy (Atwater General Factors)" and
    # "Energy (Atwater Specific Factors)" — these must collapse to "Energy" so
    # the resulting column is always "Energy (kcal)", matching the lookup table.
    energy_mask = long_df["nutrient_name"].str.contains("Energy", case=False, na=False)
    long_df.loc[energy_mask, "nutrient_name"] = "Energy"

    # Build column name = "Nutrient name (unit)"
    long_df["col"] = long_df["nutrient_name"] + " (" + long_df["unit_name"].fillna("") + ")"

    # Drop kJ energy rows — redundant with kcal and pollutes the nutrient output.
    long_df = long_df[long_df["col"] != "Energy (kJ)"]

    # Deduplicate: for each (fdc_id, col) keep the first row
    # (files were concatenated in priority order so first = best source)
    long_df = long_df.drop_duplicates(subset=["fdc_id", "col"], keep="first")

    log.info("  pivoting %d nutrient rows ...", len(long_df))
    wide = (
        long_df.set_index(["fdc_id", "col"])["amount"]
        .unstack(level="col")
        .reset_index()
    )
    wide.sort_values("fdc_id", inplace=True)
    wide.reset_index(drop=True, inplace=True)

    out_path = data_dir / "foods_wide.csv"
    wide.to_csv(out_path, index=False)
    log.info(
        "  wrote %s  (%d foods × %d nutrients)",
        out_path,
        len(wide),
        wide.shape[1] - 1,
    )
    return wide


# ---------------------------------------------------------------------------
# Step D — food_embeddings_mpnet.npy
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Harmonise food description text before embedding.

    Must stay identical to the same function in foodscribe/retrieval/openai_retriever.py.
    Removes parentheses, percent signs, and lowercases everything.
    """
    import re
    text = text.lower()
    text = re.sub(r"[()%]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_embeddings(meta: pd.DataFrame, data_dir: Path, batch_size: int = 256) -> None:
    """
    Encode food descriptions with all-mpnet-base-v2 and save L2-normalised
    float32 embeddings as food_embeddings_mpnet.npy.
    Row order matches food_metadata.csv row order.
    """
    log.info("Step D — building food_embeddings_mpnet.npy ...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    descriptions = [_normalise_text(d) for d in meta["description"].fillna("").tolist()]
    n = len(descriptions)
    log.info("  encoding %d descriptions with all-mpnet-base-v2 ...", n)

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    out_path = data_dir / "food_embeddings_mpnet.npy"
    np.save(out_path, embeddings)
    log.info("  wrote %s  shape=%s  dtype=%s", out_path, embeddings.shape, embeddings.dtype)


# ---------------------------------------------------------------------------
# Step D (alt) — food_embeddings_openai.npy
# ---------------------------------------------------------------------------

def build_embeddings_openai(meta: pd.DataFrame, data_dir: Path, batch_size: int = 512) -> None:
    """
    Encode food descriptions with OpenAI text-embedding-3-large (3072-dim) and
    save L2-normalised float32 embeddings as food_embeddings_openai.npy.
    Row order matches food_metadata.csv row order.
    Requires OPENAI_API_KEY in the environment.
    """
    import os
    log.info("Step D — building food_embeddings_openai.npy (text-embedding-3-large) ...")
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    descriptions = [_normalise_text(d) for d in meta["description"].fillna("").tolist()]
    n = len(descriptions)
    dim = 3072  # text-embedding-3-large output dimension
    log.info("  embedding %d descriptions in batches of %d ...", n, batch_size)

    embeddings = np.zeros((n, dim), dtype=np.float32)
    for i in range(0, n, batch_size):
        batch = descriptions[i : i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-large", input=batch)
        for j, emb_obj in enumerate(response.data):
            embeddings[i + j] = emb_obj.embedding
        log.info("  %d / %d done", min(i + batch_size, n), n)

    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = (embeddings / norms).astype(np.float32)

    out_path = data_dir / "food_embeddings_openai.npy"
    np.save(out_path, embeddings)
    log.info("  wrote %s  shape=%s  dtype=%s", out_path, embeddings.shape, embeddings.dtype)


# ---------------------------------------------------------------------------
# Step E — NHANES integration
# ---------------------------------------------------------------------------

# Maps NHANES DR1I* column names → standard nutrient names used in foods_wide.csv
_NHANES_NUTRIENT_MAP: dict[str, str] = {
    "DR1IKCAL": "Energy (kcal)",
    "DR1IPROT": "Protein (g)",
    "DR1ICARB": "Carbohydrate, by difference (g)",
    "DR1ISUGR": "Sugars, Total (g)",
    "DR1IFIBE": "Fiber, total dietary (g)",
    "DR1ITFAT": "Total lipid (fat) (g)",
    "DR1ISFAT": "Fatty acids, total saturated (g)",
    "DR1IMFAT": "Fatty acids, total monounsaturated (g)",
    "DR1IPFAT": "Fatty acids, total polyunsaturated (g)",
    "DR1ICHOL": "Cholesterol (mg)",
    "DR1ICALC": "Calcium, Ca (mg)",
    "DR1IPHOS": "Phosphorus, P (mg)",
    "DR1IMAGN": "Magnesium, Mg (mg)",
    "DR1IIRON": "Iron, Fe (mg)",
    "DR1IZINC": "Zinc, Zn (mg)",
    "DR1ICOPP": "Copper, Cu (mg)",
    "DR1ISODI": "Sodium, Na (mg)",
    "DR1IPOTA": "Potassium, K (mg)",
    "DR1ISELE": "Selenium, Se (µg)",
    "DR1IVC":   "Vitamin C, total ascorbic acid (mg)",
    "DR1IVB1":  "Thiamin (mg)",
    "DR1IVB2":  "Riboflavin (mg)",
    "DR1INIAC": "Niacin (mg)",
    "DR1IVB6":  "Vitamin B-6 (mg)",
    "DR1IVB12": "Vitamin B-12 (µg)",
    "DR1IVD":   "Vitamin D (D2 + D3) (µg)",
    "DR1IVK":   "Vitamin K (phylloquinone) (µg)",
    "DR1IATOC": "Vitamin E (alpha-tocopherol) (mg)",
    "DR1IRET":  "Retinol (µg)",
    "DR1IVARA": "Vitamin A, RAE (µg)",
    "DR1IBCAR": "Carotene, beta (µg)",
    "DR1IACAR": "Carotene, alpha (µg)",
    "DR1ICRYP": "Cryptoxanthin, beta (µg)",
    "DR1ILYCO": "Lycopene (µg)",
    "DR1ILZ":   "Lutein + zeaxanthin (µg)",
    "DR1IFOLA": "Folate, total (µg)",
    "DR1IFA":   "Folic acid (µg)",
    "DR1IFDFE": "Folate, DFE (µg)",
    "DR1ICHL":  "Choline, total (mg)",
    "DR1ICAFF": "Caffeine (mg)",
    "DR1ITHEO": "Theobromine (mg)",
    "DR1IALCO": "Alcohol, ethyl (g)",
    "DR1IMOIS": "Water (g)",
}

# Anomalous sentinel value used in NHANES for missing data
_NHANES_SENTINEL = 5.397605346934028e-79

# FNDDS food code prefix (first 2 digits) → USDA food category
_FNDDS_PREFIX_TO_CATEGORY: dict[str, str] = {
    "11": "Dairy and Egg Products",
    "12": "Dairy and Egg Products",
    "13": "Dairy and Egg Products",
    "14": "Dairy and Egg Products",
    "21": "Beef Products",
    "22": "Pork Products",
    "23": "Lamb, Veal, and Game Products",
    "24": "Poultry Products",
    "25": "Sausages and Luncheon Meats",
    "26": "Finfish and Shellfish Products",
    "27": "Meals, Entrees, and Side Dishes",
    "28": "Meals, Entrees, and Side Dishes",
    "29": "Meals, Entrees, and Side Dishes",
    "31": "Legumes and Legume Products",
    "32": "Dairy and Egg Products",
    "33": "Dairy and Egg Products",
    "41": "Legumes and Legume Products",
    "42": "Nuts and Seed Products",
    "43": "Nuts and Seed Products",
    "51": "Baked Products",
    "52": "Baked Products",
    "53": "Baked Products",
    "54": "Snacks",
    "55": "Baked Products",
    "56": "Cereal Grains and Pasta",
    "57": "Breakfast Cereals",
    "58": "Fast Foods",
    "61": "Fruits and Fruit Juices",
    "62": "Fruits and Fruit Juices",
    "63": "Fruits and Fruit Juices",
    "64": "Fruits and Fruit Juices",
    "67": "Baby Foods",
    "71": "Vegetables and Vegetable Products",
    "72": "Vegetables and Vegetable Products",
    "73": "Vegetables and Vegetable Products",
    "74": "Vegetables and Vegetable Products",
    "75": "Vegetables and Vegetable Products",
    "76": "Baby Foods",
    "77": "Meals, Entrees, and Side Dishes",
    "78": "Beverages",
    "81": "Fats and Oils",
    "82": "Fats and Oils",
    "83": "Fats and Oils",
    "89": "Spices and Herbs",
    "91": "Sweets",
    "92": "Beverages",
    "93": "Beverages",
    "94": "Beverages",
    "95": "Beverages",
}


def build_nhanes(nhanes_dir: Path, existing_fdc_ids: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse NHANES DR1IFF_J_with_descriptions.csv and return two DataFrames
    compatible with food_metadata.csv and foods_wide.csv schemas.

    Only foods whose DR1IFDCD code is NOT already in existing_fdc_ids are returned
    (USDA data takes priority for overlapping food codes).

    Nutrients are normalised from per-serving to per-100g by dividing each
    nutrient value by DR1IGRMS and multiplying by 100, then averaged across
    all intake records for the same food code.

    Returns:
        nhanes_meta  — columns: fdc_id, description, food_category, data_type
        nhanes_wide  — columns: fdc_id + mapped nutrient columns
    """
    log.info("Step E — integrating NHANES data ...")

    nhanes_file = nhanes_dir / "DR1IFF_J_with_descriptions.csv"
    if not nhanes_file.exists():
        log.warning("  DR1IFF_J_with_descriptions.csv not found in %s — skipping NHANES", nhanes_dir)
        return pd.DataFrame(), pd.DataFrame()

    log.info("  reading %s ...", nhanes_file)
    needed_cols = {"DR1IFDCD", "DRXFCLD", "DRXFCSD", "DR1IGRMS"} | set(_NHANES_NUTRIENT_MAP.keys())
    df = pd.read_csv(nhanes_file, usecols=lambda c: c in needed_cols, low_memory=False)

    # Replace anomalous sentinel with NaN
    df.replace(_NHANES_SENTINEL, np.nan, inplace=True)

    # Drop rows without a food code or valid gram weight
    df = df.dropna(subset=["DR1IFDCD", "DR1IGRMS"])
    df = df[df["DR1IGRMS"] > 0]

    df["DR1IFDCD"] = df["DR1IFDCD"].astype(int)

    # Exclude foods already covered by USDA data
    new_mask = ~df["DR1IFDCD"].isin(existing_fdc_ids)
    df_new = df[new_mask].copy()
    n_overlap = (~new_mask).sum()
    log.info(
        "  %d total NHANES records | %d overlap with USDA (skipped) | %d new records",
        len(df), n_overlap, len(df_new),
    )

    if df_new.empty:
        log.info("  No new NHANES foods to add.")
        return pd.DataFrame(), pd.DataFrame()

    # Normalise nutrient columns to per-100g
    nutrient_cols = [c for c in _NHANES_NUTRIENT_MAP if c in df_new.columns]
    for col in nutrient_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce") / df_new["DR1IGRMS"] * 100

    # Aggregate: one row per unique food code
    # Description: most common long description for that food code
    desc_map = (
        df_new.groupby("DR1IFDCD")["DRXFCLD"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    )
    short_map = (
        df_new.groupby("DR1IFDCD")["DRXFCSD"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    )
    nutrient_means = df_new.groupby("DR1IFDCD")[nutrient_cols].mean()

    # Map FNDDS code prefix → food category
    fdc_ids = desc_map.index.tolist()
    categories = [
        _FNDDS_PREFIX_TO_CATEGORY.get(str(fdc_id)[:2], "NHANES")
        for fdc_id in fdc_ids
    ]

    # Build metadata DataFrame
    nhanes_meta = pd.DataFrame({
        "fdc_id":        fdc_ids,
        "description":   desc_map.values,
        "food_category": categories,
        "data_type":     "nhanes",
    }).reset_index(drop=True)

    # Build wide DataFrame: rename columns to standard nutrient names
    nhanes_wide = nutrient_means.reset_index().rename(columns={"DR1IFDCD": "fdc_id"})
    nhanes_wide.rename(columns=_NHANES_NUTRIENT_MAP, inplace=True)

    log.info(
        "  NHANES: %d unique new foods, %d nutrient columns",
        len(nhanes_meta),
        len(nutrient_cols),
    )

    # Store short description for use as subcategory in categories step
    nhanes_meta["_short_desc"] = short_map.values

    return nhanes_meta, nhanes_wide


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build FoodScribe data files from raw USDA long-format CSVs."
    )
    p.add_argument(
        "--usda-dir",
        default="USDA_data/",
        help="Directory containing foundation_long.csv, legacy_long.csv, survey_long.csv",
    )
    p.add_argument(
        "--data-dir",
        default="data/",
        help="Output directory for generated data files",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip the embedding step (steps A-C only)",
    )
    p.add_argument(
        "--embedder",
        choices=["mpnet", "openai"],
        default="mpnet",
        help="Embedding model to use: mpnet (local, default) or openai (text-embedding-3-large)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256 for mpnet, 512 recommended for openai)",
    )
    p.add_argument(
        "--nhanes-dir",
        default=None,
        help="Directory containing DR1IFF_J_with_descriptions.csv (NHANES integration, optional)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    usda_dir = Path(args.usda_dir)
    data_dir = Path(args.data_dir)

    if not usda_dir.exists():
        log.error("USDA data directory not found: %s", usda_dir)
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # A — metadata
    meta = build_metadata(usda_dir, data_dir)

    # B — categories
    build_categories(meta, data_dir)

    # C — wide nutrient table
    wide = build_foods_wide(usda_dir, data_dir, set(meta["fdc_id"].tolist()))

    # E — NHANES integration (optional)
    if args.nhanes_dir:
        nhanes_dir = Path(args.nhanes_dir)
        nhanes_meta, nhanes_wide = build_nhanes(nhanes_dir, set(meta["fdc_id"].tolist()))

        if not nhanes_meta.empty:
            # Append NHANES foods to food_metadata.csv
            short_descs = nhanes_meta.pop("_short_desc")
            combined_meta = pd.concat([meta, nhanes_meta], ignore_index=True)
            combined_meta.to_csv(data_dir / "food_metadata.csv", index=False)
            log.info(
                "  food_metadata.csv updated: %d USDA + %d NHANES = %d total foods",
                len(meta), len(nhanes_meta), len(combined_meta),
            )

            # Append NHANES foods to food_categories.csv
            nhanes_cats = nhanes_meta[["fdc_id", "description", "food_category", "data_type"]].copy()
            nhanes_cats.rename(columns={"food_category": "category"}, inplace=True)
            nhanes_cats["subcategory"] = short_descs.values
            nhanes_cats = nhanes_cats[["fdc_id", "description", "category", "subcategory", "data_type"]]
            existing_cats = pd.read_csv(data_dir / "food_categories.csv")
            combined_cats = pd.concat([existing_cats, nhanes_cats], ignore_index=True)
            combined_cats.to_csv(data_dir / "food_categories.csv", index=False)
            log.info("  food_categories.csv updated: %d total rows", len(combined_cats))

            # Append NHANES foods to foods_wide.csv (new rows; fill missing nutrient cols with NaN)
            combined_wide = pd.concat([wide, nhanes_wide], ignore_index=True, sort=False)
            combined_wide.to_csv(data_dir / "foods_wide.csv", index=False)
            log.info(
                "  foods_wide.csv updated: %d total foods × %d nutrient columns",
                len(combined_wide), combined_wide.shape[1] - 1,
            )

            # Use combined metadata for embedding step
            meta = combined_meta

    # D — embeddings (optional)
    if args.skip_embeddings:
        log.info("Step D — skipped (--skip-embeddings)")
    elif args.embedder == "openai":
        build_embeddings_openai(meta, data_dir, batch_size=args.batch_size or 512)
    else:
        build_embeddings(meta, data_dir, batch_size=args.batch_size)

    log.info("Done. All data files written to %s/", data_dir)


if __name__ == "__main__":
    main()
