"""
One-off script to build all data/ files required by FoodScribe from raw USDA long-format CSVs.

Input files (--usda-dir):
    foundation_long.csv   — USDA Foundation foods, long format
    legacy_long.csv       — USDA SR Legacy foods, long format
    survey_long.csv       — USDA Survey/FNDDS foods, long format

    All three share schema:
        fdc_id, data_type, ndb_number, description, food_category,
        nutrient_name, amount, unit_name, ...

Output files (--data-dir):
    food_metadata.csv          fdc_id, description, food_category, data_type
    food_categories.csv        fdc_id, description, category, subcategory, data_type
    foods_wide.csv             fdc_id + one column per nutrient (per 100 g)
    food_embeddings_mpnet.npy  (N, 768) float32, L2-normalised

Usage:
    python scripts/build_data.py --usda-dir ../USDA_data/ --data-dir data/
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

    descriptions = meta["description"].fillna("").tolist()
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
    descriptions = meta["description"].fillna("").tolist()
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
    build_foods_wide(usda_dir, data_dir, set(meta["fdc_id"].tolist()))

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
