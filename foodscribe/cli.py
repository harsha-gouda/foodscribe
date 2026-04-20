"""FoodScribe CLI — entry point."""
from __future__ import annotations

import os

# Must be set before any torch/sentence-transformers import to suppress
# duplicate OpenMP DLL warning on Windows + Anaconda.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="FoodScribe: USDA-grounded meal nutrient analyser.")

DATA_DIR = Path(os.environ.get("FOODSCRIBE_DATA_DIR", "data/"))

_VALID_PROVIDERS = ("anthropic", "openai", "deepseek", "gemini")
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _set_env_var(key: str, value: str) -> None:
    """Write or update a key=value line in the project .env file."""
    lines: list[str] = _ENV_FILE.read_text(encoding="utf-8").splitlines() if _ENV_FILE.exists() else []
    prefix = f"{key}="
    new_line = f"{key}={value}"
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = new_line
            _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    lines.append(new_line)
    _ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command("use-provider")
def use_provider(
    provider: str = typer.Argument(..., help="Provider to use by default: anthropic | openai | deepseek | gemini"),
) -> None:
    """Set the default LLM provider (saved to .env — no need to pass --provider on every command)."""
    p = provider.lower()
    if p not in _VALID_PROVIDERS:
        typer.echo(f"[error] Unknown provider {p!r}. Choose from: {', '.join(_VALID_PROVIDERS)}", err=True)
        raise typer.Exit(1)
    _set_env_var("FOODSCRIBE_LLM_PROVIDER", p)
    typer.echo(f"Default provider set to: {p}  (saved to {_ENV_FILE})")


def _make_retriever(data_dir: Path, top_k: int, foundation_threshold: float = 0.70):
    """Return OpenAIRetriever if the OpenAI index exists, else MPNetRetriever."""
    openai_index = data_dir / "food_embeddings_openai.npy"
    if openai_index.exists():
        from foodscribe.retrieval.openai_retriever import OpenAIRetriever
        return OpenAIRetriever(data_dir=data_dir, top_k=top_k, foundation_threshold=foundation_threshold)
    from foodscribe.retrieval.mpnet_retriever import MPNetRetriever
    return MPNetRetriever(data_dir=data_dir, top_k=top_k, foundation_threshold=foundation_threshold)


def _make_pipeline(
    provider: str | None,
    model: str | None,
    top_k: int,
    data_dir: Path,
    foundation_threshold: float = 0.70,
):
    from foodscribe.llm.client import LLMClient
    from foodscribe.nutrients.categories import CategoryLookup
    from foodscribe.nutrients.lookup import NutrientLookup
    from foodscribe.analysis.stats import MealAnalyser

    llm = LLMClient(provider=provider, model=model)
    retriever = _make_retriever(data_dir, top_k, foundation_threshold)
    cat_lookup = CategoryLookup(data_dir=data_dir)
    nut_lookup = NutrientLookup(data_dir=data_dir, category_lookup=cat_lookup)
    analyser = MealAnalyser()
    return llm, retriever, nut_lookup, analyser


@app.command()
def parse(
    meal_text: str = typer.Argument(..., help="Free-text meal description"),
    provider: str = typer.Option(None, envvar="FOODSCRIBE_LLM_PROVIDER",
                                  help="LLM provider: anthropic | openai | deepseek | gemini"),
    model: str = typer.Option(None, help="Model name override"),
    top_k: int = typer.Option(3, help="Retrieval candidates per item"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON instead of table"),
    plot: bool = typer.Option(False, "--plot", help="Show matplotlib plots (all four)"),
    save_plots: str = typer.Option(None, help="Save plots as PNG to this directory"),
    show_category: bool = typer.Option(False, "--show-category", help="Add Category/Subcategory columns"),
    all_nutrients: bool = typer.Option(False, "--all-nutrients", help="Show full micronutrient panel for the meal"),
    foundation_threshold: float = typer.Option(0.70, "--foundation-threshold",
        help="Min cosine score to prefer a Foundation food (0=off, 1=Foundation-only)"),
    data_dir: str = typer.Option(None, envvar="FOODSCRIBE_DATA_DIR", help="Data directory path"),
) -> None:
    """Parse a meal description and return USDA nutrient profile."""
    ddir = Path(data_dir) if data_dir else DATA_DIR
    llm, retriever, nut_lookup, analyser = _make_pipeline(provider, model, top_k, ddir,
                                                           foundation_threshold=foundation_threshold)

    typer.echo(f"Provider: {llm.provider} ({llm.model})")

    # Stage 1: parse meal text
    try:
        items = llm.parse_meal(meal_text)
    except Exception as exc:
        typer.echo(f"[error] LLM parsing failed: {exc}", err=True)
        raise typer.Exit(1)

    if not items:
        rejection = {"item": None, "error": "no_food_items_found", "input": meal_text}
        if json_out:
            typer.echo(json.dumps([rejection], indent=2))
        else:
            typer.echo(f"[rejected] {meal_text!r} — no_food_items_found", err=True)
        raise typer.Exit(1)

    if json_out:
        import dataclasses
        typer.echo(json.dumps([dataclasses.asdict(i) for i in items], indent=2))
        return

    # Stage 2: retrieve fdc_ids
    # Pass meal_text as context so MPNet can disambiguate by preparation style
    queries = [item.item for item in items]
    contexts = [meal_text] * len(queries)
    batch_results = retriever.retrieve_batch(queries, top_k=top_k, contexts=contexts)

    # Stage 3: nutrient lookup
    from foodscribe.nutrients.lookup import NutrientRow
    rows: list[NutrientRow] = []
    for item, candidates in zip(items, batch_results):
        if not candidates:
            typer.echo(f"[warning] No match found for '{item.item}'", err=True)
            continue
        best = candidates[0]
        if item.grams:
            row = nut_lookup.get_scaled(best.fdc_id, item.grams)
        else:
            row = nut_lookup.get(best.fdc_id)
        if row is None:
            typer.echo(f"[warning] No nutrient data for fdc_id {best.fdc_id} ('{best.description}')", err=True)
            continue
        # Attach retrieval metadata for display
        row._confidence = item.confidence
        row._grams = item.grams
        rows.append(row)

    if not rows:
        typer.echo("[error] No nutrient data found for any items.", err=True)
        raise typer.Exit(1)

    # Stage 4: summarise + display
    ingredient_parts = []
    for item in items:
        part = item.item
        if item.grams:
            part += f" ({item.grams:.0f}g)"
        ingredient_parts.append(part)
    typer.echo(f"Identified: {' · '.join(ingredient_parts)}\n")

    summary = analyser.summarise(rows)
    analyser.print_table(rows, summary, show_category=show_category, show_all_nutrients=all_nutrients)

    if plot or save_plots:
        plots_dir = Path(save_plots) if save_plots else None
        if plots_dir:
            plots_dir.mkdir(parents=True, exist_ok=True)

        def _path(name: str) -> str | None:
            return str(plots_dir / name) if plots_dir else None

        analyser.plot_macros_pie(summary, save_path=_path("macros_pie.png"))
        analyser.plot_energy_distribution(rows, save_path=_path("energy_dist.png"))
        analyser.plot_nutrient_bars(rows, save_path=_path("nutrient_bars.png"))
        analyser.plot_category_breakdown(rows, save_path=_path("category_breakdown.png"))

        if save_plots:
            typer.echo(f"Plots saved to {plots_dir}/")


@app.command()
def categories(
    fdc_id: int = typer.Option(None, help="Look up category for a specific fdc_id"),
    list_all: bool = typer.Option(False, "--list", help="Print all available top-level categories"),
    filter_cat: str = typer.Option(None, "--filter", help="Show all foods matching a category"),
    data_dir: str = typer.Option(None, envvar="FOODSCRIBE_DATA_DIR"),
) -> None:
    """Look up USDA food categories."""
    from foodscribe.nutrients.categories import CategoryLookup

    ddir = Path(data_dir) if data_dir else DATA_DIR
    lookup = CategoryLookup(data_dir=ddir)

    if fdc_id is not None:
        fc = lookup.get(fdc_id)
        if fc is None:
            typer.echo(f"fdc_id {fdc_id} not found in food_categories.csv")
            raise typer.Exit(1)
        typer.echo(f"fdc_id      : {fc.fdc_id}")
        typer.echo(f"Description : {fc.description}")
        typer.echo(f"Category    : {fc.category}")
        typer.echo(f"Subcategory : {fc.subcategory or '—'}")
        typer.echo(f"Data type   : {fc.data_type}")
    elif list_all:
        for cat in lookup.list_categories():
            typer.echo(cat)
    elif filter_cat:
        matches = lookup.filter_by_category(filter_cat)
        if not matches:
            typer.echo(f"No foods found matching '{filter_cat}'")
        else:
            for fc in matches[:50]:
                typer.echo(f"{fc.fdc_id:>8}  {fc.category:<35}  {fc.description[:60]}")
            if len(matches) > 50:
                typer.echo(f"... ({len(matches)} total — showing first 50)")
    else:
        typer.echo("Use --fdc-id, --list, or --filter. Run with --help for details.")
        raise typer.Exit(1)


@app.command()
def analyse(
    meal: str = typer.Option(None, help="Meal description string (or reads from stdin)"),
    csv: str = typer.Option(None, help="CSV with a 'meal' column — aggregate stats"),
    provider: str = typer.Option(None, envvar="FOODSCRIBE_LLM_PROVIDER"),
    data_dir: str = typer.Option(None, envvar="FOODSCRIBE_DATA_DIR"),
) -> None:
    """Analyse a meal or batch of meals from a CSV."""
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    ddir = Path(data_dir) if data_dir else DATA_DIR
    llm, retriever, nut_lookup, analyser = _make_pipeline(provider, None, 3, ddir)

    def _run_meal(text: str):
        items = llm.parse_meal(text)
        queries = [i.item for i in items]
        batch = retriever.retrieve_batch(queries, contexts=[text] * len(queries))
        rows = []
        for item, cands in zip(items, batch):
            if not cands:
                continue
            row = nut_lookup.get_scaled(cands[0].fdc_id, item.grams) if item.grams else nut_lookup.get(cands[0].fdc_id)
            if row:
                rows.append(row)
        return rows

    if csv:
        df = pd.read_csv(csv)
        if "meal" not in df.columns:
            typer.echo("[error] CSV must have a 'meal' column", err=True)
            raise typer.Exit(1)
        summaries = []
        for meal_text in df["meal"]:
            rows = _run_meal(str(meal_text))
            if rows:
                summaries.append(analyser.summarise(rows))

        if not summaries:
            typer.echo("No meals could be analysed.")
            raise typer.Exit(1)

        metrics = {
            "energy_kcal": [s.total_energy_kcal for s in summaries],
            "protein_g":   [s.total_protein_g for s in summaries],
            "carb_g":      [s.total_carb_g for s in summaries],
            "fat_g":       [s.total_fat_g for s in summaries],
        }
        typer.echo(f"\nAggregate stats across {len(summaries)} meals:")
        for key, vals in metrics.items():
            arr = np.array(vals)
            typer.echo(f"  {key:15s}  mean={arr.mean():.1f}  sd={arr.std():.1f}")

        # Plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (key, vals) in zip(axes.flatten(), metrics.items()):
            ax.boxplot(vals); ax.set_title(key); ax.set_ylabel("value")
        plt.suptitle("Macro Distribution Across Meals"); plt.tight_layout()
        plt.show()

    elif meal:
        rows = _run_meal(meal)
        if not rows:
            typer.echo("[error] No nutrient data found.", err=True)
            raise typer.Exit(1)
        summary = analyser.summarise(rows)
        analyser.print_table(rows, summary, show_category=True)
    else:
        if not sys.stdin.isatty():
            meal_text = sys.stdin.read().strip()
            rows = _run_meal(meal_text)
            summary = analyser.summarise(rows)
            analyser.print_table(rows, summary, show_category=True)
        else:
            typer.echo("Provide --meal TEXT or --csv PATH (or pipe text via stdin).")
            raise typer.Exit(1)


@app.command("batch-parse")
def batch_parse(
    input_file: str = typer.Argument(None, help="CSV file to process (filename or path). Omit to process all CSVs in --input-dir."),
    input_dir: str = typer.Option("input", help="Folder containing input CSV files (used when no input_file is given)"),
    output_dir: str = typer.Option("output", help="Base output folder; a run subfolder is created automatically"),
    run_id: str = typer.Option(None, help="Run subfolder name (default: run_YYYYMMDD_HHMMSS). Pass an existing run_id to resume."),
    meal_col: str = typer.Option("meal", help="Name of the meal text column in input CSVs"),
    provider: str = typer.Option(None, envvar="FOODSCRIBE_LLM_PROVIDER"),
    model: str = typer.Option(None, help="Model name override"),
    limit: int = typer.Option(200, help="Max rows to process per run (0 = no limit). Re-run with same --run-id to continue."),
) -> None:
    """Step 1 — LLM only: parse meals into ingredients and save to *_parsed.csv.

    Each run creates output/<run_id>/ automatically (e.g. output/run_20241201_143022/).
    To resume an interrupted run, pass --run-id run_20241201_143022.

    Output columns: row, <original columns>, meal, ingredient, qty, unit, grams, confidence
    Run batch-nutrients --run-id <same id> on the output to get USDA nutrient profiles.
    """
    import json
    import pandas as pd
    from datetime import datetime

    _parse_start = datetime.now()

    # Resolve which CSV files to process
    if input_file:
        p = Path(input_file)
        if not p.exists():
            # Try relative to input_dir
            p = Path(input_dir) / input_file
        if not p.exists():
            typer.echo(f"[error] File not found: {input_file}", err=True)
            raise typer.Exit(1)
        csv_files = [p]
    else:
        in_dir = Path(input_dir)
        if not in_dir.exists():
            typer.echo(f"[error] Input folder not found: {in_dir}", err=True)
            raise typer.Exit(1)
        csv_files = sorted(in_dir.glob("*.csv"))
        if not csv_files:
            typer.echo(f"[error] No CSV files found in {in_dir}", err=True)
            raise typer.Exit(1)

    _run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(output_dir) / _run_id

    out_dir.mkdir(parents=True, exist_ok=True)

    from foodscribe.llm.client import LLMClient
    llm = LLMClient(provider=provider, model=model)
    typer.echo(f"Provider : {llm.provider} ({llm.model})")
    typer.echo(f"Run ID   : {_run_id}  (pass --run-id {_run_id} to resume)")
    batch_size = limit if limit > 0 else None
    source_desc = csv_files[0] if input_file else f"{Path(input_dir)}/"
    typer.echo(f"Parsing {len(csv_files)} file(s) from {source_desc} -> {out_dir}/  (limit={batch_size or 'none'})\n")

    _parse_file_stats: list[dict] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if meal_col not in df.columns:
            typer.echo(f"[skip] {csv_path.name} — no '{meal_col}' column", err=True)
            continue

        out_path = out_dir / f"{csv_path.stem}_parsed.csv"

        # Resume: find which row numbers were already processed
        done_rows: set[int] = set()
        if out_path.exists():
            existing = pd.read_csv(out_path)
            done_rows = set(existing["row"].tolist())
            typer.echo(f"[{csv_path.name}] {len(df)} rows  ({len(done_rows)} already done, resuming)")
        else:
            typer.echo(f"[{csv_path.name}] {len(df)} rows")

        # Extra columns to carry through (everything except the meal column)
        extra_cols = [c for c in df.columns if c != meal_col]

        records = []
        processed = 0

        for idx, (_, row_data) in enumerate(df.iterrows(), start=1):
            if idx in done_rows:
                continue
            if batch_size and processed >= batch_size:
                remaining = len(df) - max(done_rows, default=0) - processed
                typer.echo(f"  [limit reached] {remaining} row(s) still pending — re-run to continue")
                break

            meal_text = str(row_data[meal_col]).strip()
            if not meal_text:
                continue

            # Collect extra column values for this row
            extra = {c: row_data[c] for c in extra_cols}

            try:
                items = llm.parse_meal(meal_text)
            except Exception as exc:
                typer.echo(f"  row {idx}: [error] {exc}", err=True)
                processed += 1
                continue

            if not items:
                typer.echo(f"  row {idx}: [skip] no food items recognised")
                processed += 1
                continue

            for item in items:
                records.append({
                    "row":        idx,
                    **extra,
                    "meal":       meal_text,
                    "ingredient": item.item,
                    "qty":        item.qty,
                    "unit":       item.unit,
                    "grams":      item.grams,
                    "confidence": item.confidence,
                })
            typer.echo(f"  row {idx}: {len(items)} ingredient(s)")
            processed += 1

        if records:
            new_df = pd.DataFrame(records)
            if out_path.exists():
                new_df.to_csv(out_path, mode="a", header=False, index=False)
            else:
                new_df.to_csv(out_path, index=False)
            typer.echo(f"  -> {out_path}  (+{len(records)} rows)")

        _parse_file_stats.append({
            "file": csv_path.name,
            "total_rows": len(df),
            "rows_processed_this_run": processed,
            "rows_done_previously": len(done_rows),
        })

    # Write / update run_metadata.json
    _meta_path = out_dir / "run_metadata.json"
    _meta: dict = json.loads(_meta_path.read_text()) if _meta_path.exists() else {"run_id": _run_id}
    _meta["batch_parse"] = {
        "started_at":        _parse_start.isoformat(timespec="seconds"),
        "completed_at":      datetime.now().isoformat(timespec="seconds"),
        "duration_seconds":  round((datetime.now() - _parse_start).total_seconds(), 1),
        "provider":          llm.provider,
        "model":             llm.model,
        "row_limit_per_run": batch_size or "none",
        "files":             _parse_file_stats,
    }
    _meta_path.write_text(json.dumps(_meta, indent=2))
    typer.echo(f"  -> {_meta_path}")

    typer.echo("\nParsing complete.  Run 'foodscribe batch-nutrients' next.")


@app.command("batch-nutrients")
def batch_nutrients(
    input_file: str = typer.Argument(None, help="Specific *_parsed.csv file to process (filename or path). Omit to use --run-id / auto-detect."),
    output_dir: str = typer.Option("output", help="Base output folder (same as used for batch-parse)"),
    run_id: str = typer.Option(None, help="Run subfolder to process (default: latest run_* folder found)"),
    top_k: int = typer.Option(3, help="Retrieval candidates per item"),
    foundation_threshold: float = typer.Option(0.70, "--foundation-threshold",
        help="Min cosine score to prefer a Foundation food (0=off, 1=Foundation-only)"),
    data_dir: str = typer.Option(None, envvar="FOODSCRIBE_DATA_DIR"),
) -> None:
    """Step 2 — Retrieval only: take *_parsed.csv files and produce nutrient profiles.

    Reads *_parsed.csv from output/<run_id>/ (auto-detects the latest run folder).
    Writes <name>_summary.csv and <name>_detail.csv to the same run folder.
    Pass a specific file path as the first argument, or use --run-id to target a run folder.
    """
    import json
    import pandas as pd
    from datetime import datetime as _dt
    from foodscribe.nutrients.categories import CategoryLookup
    from foodscribe.nutrients.lookup import NutrientLookup
    from foodscribe.analysis.stats import MealAnalyser, _f, _s

    # Resolve parsed files and output folder
    if input_file:
        p = Path(input_file)
        if not p.exists():
            typer.echo(f"[error] File not found: {input_file}", err=True)
            raise typer.Exit(1)
        parsed_files = [p]
        in_dir = p.parent
        out_dir = p.parent
        typer.echo(f"Input file: {p}")
    else:
        base_dir = Path(output_dir)
        if run_id:
            in_dir = base_dir / run_id
        else:
            run_dirs = sorted(base_dir.glob("run_*"), reverse=True)
            in_dir = run_dirs[0] if run_dirs else base_dir
        out_dir = in_dir
        typer.echo(f"Run folder: {in_dir}")

        if not in_dir.exists():
            typer.echo(f"[error] Input folder not found: {in_dir}", err=True)
            raise typer.Exit(1)

        parsed_files = sorted(in_dir.glob("*_parsed.csv"))
        if not parsed_files:
            typer.echo(f"[error] No *_parsed.csv files found in {in_dir}", err=True)
            raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    ddir = Path(data_dir) if data_dir else DATA_DIR
    _nutrients_start = _dt.now()

    typer.echo("Loading retrieval index...")
    retriever = _make_retriever(ddir, top_k, foundation_threshold)
    cat_lookup = CategoryLookup(data_dir=ddir)
    nut_lookup = NutrientLookup(data_dir=ddir, category_lookup=cat_lookup)
    analyser = MealAnalyser()
    typer.echo(f"Processing {len(parsed_files)} file(s) from {in_dir}/ -> {out_dir}/\n")

    _nutrients_file_stats: list[dict] = []

    for parsed_path in parsed_files:
        df = pd.read_csv(parsed_path)
        required = {"row", "meal", "ingredient", "grams", "confidence"}
        if not required.issubset(df.columns):
            typer.echo(f"[skip] {parsed_path.name} — missing columns {required - set(df.columns)}", err=True)
            continue

        # Extra columns to carry through (everything except the pipeline columns)
        pipeline_cols = {"row", "meal", "ingredient", "qty", "unit", "grams", "confidence"}
        extra_cols = [c for c in df.columns if c not in pipeline_cols]

        typer.echo(f"[{parsed_path.name}]")
        summary_records = []
        detail_records = []

        for (row_idx, meal_text), grp in df.groupby(["row", "meal"], sort=False):
            # Take extra column values from the first ingredient row of this meal
            extra = {c: grp.iloc[0][c] for c in extra_cols}

            ingredients = grp["ingredient"].tolist()
            grams_list  = grp["grams"].tolist()
            conf_list   = grp["confidence"].tolist()

            batch_results = retriever.retrieve_batch(ingredients, top_k=top_k)
            rows = []
            for ingredient, grams, conf, cands in zip(ingredients, grams_list, conf_list, batch_results):
                if not cands:
                    continue
                grams_val = float(grams) if pd.notna(grams) else None
                row = nut_lookup.get_scaled(cands[0].fdc_id, grams_val) if grams_val is not None else nut_lookup.get(cands[0].fdc_id)
                if row:
                    row._confidence = int(conf) if pd.notna(conf) else None
                    row._grams = grams_val
                    rows.append(row)

            if not rows:
                continue

            summary = analyser.summarise(rows)
            # Sum all available nutrients across all ingredients in this meal
            all_nutrient_totals: dict[str, float] = {}
            for r in rows:
                for k, v in (r.all_nutrients or {}).items():
                    all_nutrient_totals[k] = round(all_nutrient_totals.get(k, 0.0) + v, 4)
            valid_conf = [int(c) for c in conf_list if pd.notna(c)]
            avg_conf = round(sum(valid_conf) / len(valid_conf), 2) if valid_conf else None
            summary_records.append({
                "row":        row_idx,
                **extra,
                "meal":       meal_text,
                "ingredients": "; ".join(ingredients),
                "avg_confidence": avg_conf,
                "item_count":  summary.item_count,
                "dominant_category": _s(summary.dominant_category),
                **all_nutrient_totals,
            })
            for ingredient, r in zip(ingredients, rows):
                nutrient_cols = {
                    k: round(v, 4) for k, v in (r.all_nutrients or {}).items()
                }
                detail_records.append({
                    "row":        row_idx,
                    **extra,
                    "meal":       meal_text,
                    "ingredient": ingredient,
                    "fdc_id":     r.fdc_id,
                    "usda_match": r.description,
                    "grams":      r._grams or 100,
                    "category":    _s(getattr(r, "category", None)),
                    "subcategory": _s(getattr(r, "subcategory", None)),
                    "data_type":   _s(getattr(r, "data_type", None)),
                    "confidence":  r._confidence,
                    **nutrient_cols,
                })
            typer.echo(f"  row {row_idx}: {len(rows)} item(s), {summary.total_energy_kcal:.0f} kcal")

        stem = parsed_path.stem.removesuffix("_parsed")
        if summary_records:
            p = out_dir / f"{stem}_summary.csv"
            pd.DataFrame(summary_records).to_csv(p, index=False)
            typer.echo(f"  -> {p}")
        if detail_records:
            detail_df = pd.DataFrame(detail_records)
            p = out_dir / f"{stem}_detail.csv"
            detail_df.to_csv(p, index=False)
            typer.echo(f"  -> {p}")

            # food_items.csv — rows=meals, columns=USDA food items, values=grams consumed
            # Only include fully-populated extra columns in pivot index;
            # any NaN in an index column causes groupby to silently drop those rows.
            pivot_extra_cols = [c for c in extra_cols if detail_df[c].notna().all()]
            row_index_cols = ["row"] + pivot_extra_cols + ["meal"]
            _food_items_pivot = (
                detail_df
                .groupby(row_index_cols + ["usda_match"])["grams"]
                .sum()
                .unstack(fill_value=0)
            )
            _food_items_pivot.columns.name = None
            _conflicts = set(_food_items_pivot.columns) & set(row_index_cols)
            if _conflicts:
                _food_items_pivot = _food_items_pivot.rename(columns={c: f"{c} (food)" for c in _conflicts})
            food_items_df = _food_items_pivot.reset_index()
            p = out_dir / f"{stem}_food_items.csv"
            food_items_df.to_csv(p, index=False)
            typer.echo(f"  -> {p}")

            # food_groups.csv — rows=meals, columns=food categories, values=grams consumed
            _food_groups_pivot = (
                detail_df
                .groupby(row_index_cols + ["category"])["grams"]
                .sum()
                .unstack(fill_value=0)
            )
            _food_groups_pivot.columns.name = None
            _conflicts = set(_food_groups_pivot.columns) & set(row_index_cols)
            if _conflicts:
                _food_groups_pivot = _food_groups_pivot.rename(columns={c: f"{c} (group)" for c in _conflicts})
            food_groups_df = _food_groups_pivot.reset_index()
            p = out_dir / f"{stem}_food_groups.csv"
            food_groups_df.to_csv(p, index=False)
            typer.echo(f"  -> {p}")

        _nutrients_file_stats.append({
            "file": parsed_path.name,
            "meals_processed": len(summary_records),
            "ingredients_matched": len(detail_records),
        })

    # Write / update run_metadata.json
    _meta_path = out_dir / "run_metadata.json"
    _meta: dict = json.loads(_meta_path.read_text()) if _meta_path.exists() else {"run_id": str(in_dir.name)}
    _meta["batch_nutrients"] = {
        "started_at":       _nutrients_start.isoformat(timespec="seconds"),
        "completed_at":     _dt.now().isoformat(timespec="seconds"),
        "duration_seconds": round((_dt.now() - _nutrients_start).total_seconds(), 1),
        "top_k":            top_k,
        "files":            _nutrients_file_stats,
    }
    _meta_path.write_text(json.dumps(_meta, indent=2))
    typer.echo(f"  -> {_meta_path}")

    typer.echo("\nNutrient retrieval complete.")


@app.command("ingredient-lookup")
def ingredient_lookup(
    input_file: str = typer.Argument(..., help="CSV file with an Ingredient column and a grams column"),
    ingredient_col: str = typer.Option("Ingredient", help="Column name for ingredient text"),
    grams_col: str = typer.Option("grams", help="Column name for gram weights"),
    output_file: str = typer.Option(None, help="Output path (default: <stem>_nutrients.csv next to input)"),
    top_k: int = typer.Option(3, help="Retrieval candidates per ingredient"),
    foundation_threshold: float = typer.Option(0.70, "--foundation-threshold"),
    data_dir: str = typer.Option(None, envvar="FOODSCRIBE_DATA_DIR"),
) -> None:
    """Direct ingredient-level nutrient lookup — no LLM step.

    Reads a CSV with an ingredient description column and a gram-weight column,
    runs semantic retrieval on each ingredient, and writes a nutrient profile CSV
    with one row per ingredient scaled to the given gram weight.

    All extra columns (e.g. ParticipantCode, CompFileID) are carried through.

    Examples:

      foodscribe ingredient-lookup input/food_ingredients_GutPuzzle.csv

      foodscribe ingredient-lookup input/foods.csv --ingredient-col FoodName --grams-col WeightG
    """
    import pandas as pd
    from foodscribe.nutrients.categories import CategoryLookup
    from foodscribe.nutrients.lookup import NutrientLookup
    from foodscribe.analysis.stats import _s

    in_path = Path(input_file)
    if not in_path.exists():
        typer.echo(f"[error] File not found: {in_path}", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(in_path)
    for col in (ingredient_col, grams_col):
        if col not in df.columns:
            typer.echo(f"[error] Column '{col}' not found. Available: {list(df.columns)}", err=True)
            raise typer.Exit(1)

    ddir = Path(data_dir) if data_dir else DATA_DIR
    typer.echo("Loading retrieval index...")
    retriever = _make_retriever(ddir, top_k, foundation_threshold)
    cat_lookup = CategoryLookup(data_dir=ddir)
    nut_lookup = NutrientLookup(data_dir=ddir, category_lookup=cat_lookup)

    # Extra columns to carry through
    pipeline_cols = {ingredient_col, grams_col}
    extra_cols = [c for c in df.columns if c not in pipeline_cols]

    ingredients = df[ingredient_col].tolist()
    grams_list = df[grams_col].tolist()

    typer.echo(f"Retrieving USDA matches for {len(ingredients)} ingredients...")
    batch_results = retriever.retrieve_batch(ingredients, top_k=top_k)

    detail_records = []
    for i, (ingredient, grams, cands) in enumerate(zip(ingredients, grams_list, batch_results)):
        extra = {c: df.iloc[i][c] for c in extra_cols}
        if not cands:
            typer.echo(f"  [no match] {ingredient}")
            continue
        best = cands[0]
        grams_val = float(grams) if pd.notna(grams) else None
        row = nut_lookup.get_scaled(best.fdc_id, grams_val) if grams_val is not None else nut_lookup.get(best.fdc_id)
        if not row:
            typer.echo(f"  [no nutrients] {ingredient} -> {best.description}")
            continue
        nutrient_cols = {k: round(v, 4) for k, v in (row.all_nutrients or {}).items()}
        detail_records.append({
            **extra,
            ingredient_col: ingredient,
            grams_col: grams_val or 100,
            "fdc_id":      best.fdc_id,
            "usda_match":  best.description,
            "match_score": round(best.score, 4),
            "category":    _s(getattr(row, "category", None)),
            "subcategory": _s(getattr(row, "subcategory", None)),
            "data_type":   _s(getattr(row, "data_type", None)),
            **nutrient_cols,
        })

    if not detail_records:
        typer.echo("[error] No nutrients retrieved.", err=True)
        raise typer.Exit(1)

    out_df = pd.DataFrame(detail_records)
    if output_file:
        out_path = Path(output_file)
    else:
        out_path = in_path.with_stem(in_path.stem + "_nutrients")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    typer.echo(f"\nMatched {len(detail_records)}/{len(ingredients)} ingredients")
    typer.echo(f"  -> {out_path}")


@app.command()
def aggregate(
    input_file: str = typer.Argument(..., help="CSV file to aggregate (e.g. *_summary.csv, *_food_items.csv)"),
    by: str = typer.Argument(..., help="Comma-separated columns to group by, e.g. 'Subject_ID,Date'"),
    output_file: str = typer.Option(None, help="Output path (default: auto-named next to input file)"),
    no_meal: bool = typer.Option(False, "--no-meal", help="Drop the 'meal' column before aggregating (useful for meal-level -> day-level rollup)"),
) -> None:
    """Aggregate any FoodScribe output CSV by grouping columns of interest.

    Examples:

      # Daily intake per subject from a summary file
      foodscribe aggregate results_summary.csv "Subject_ID,Date"

      # Daily food-item grams from a food_items pivot
      foodscribe aggregate results_food_items.csv "Subject_ID,Date"
    """
    import pandas as pd

    in_path = Path(input_file)
    if not in_path.exists():
        typer.echo(f"[error] File not found: {in_path}", err=True)
        raise typer.Exit(1)

    group_cols = [c.strip() for c in by.split(",") if c.strip()]
    if not group_cols:
        typer.echo("[error] --by must contain at least one column name", err=True)
        raise typer.Exit(1)

    df = pd.read_csv(in_path)

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        typer.echo(f"[error] Column(s) not found in file: {missing}", err=True)
        typer.echo(f"Available columns: {list(df.columns)}", err=True)
        raise typer.Exit(1)

    if no_meal and "meal" in df.columns:
        df = df.drop(columns=["meal"])

    # Sum all numeric columns; drop non-numeric, non-group columns
    numeric_cols = [c for c in df.select_dtypes("number").columns if c not in group_cols]
    agg_df = (
        df[group_cols + numeric_cols]
        .groupby(group_cols, sort=False)
        .sum()
        .reset_index()
    )

    # Build output path
    if output_file:
        out_path = Path(output_file)
    else:
        suffix = "_by_" + "_".join(c.replace(" ", "_") for c in group_cols)
        out_path = in_path.with_stem(in_path.stem + suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)
    typer.echo(f"Aggregated {len(df)} rows -> {len(agg_df)} groups")
    typer.echo(f"  -> {out_path}")


@app.command()
def batch(
    input_dir: str = typer.Option("input", help="Folder containing input CSV files"),
    output_dir: str = typer.Option("output", help="Folder where result CSVs are saved"),
    meal_col: str = typer.Option("meal", help="Name of the meal text column in input CSVs"),
    provider: str = typer.Option(None, envvar="FOODSCRIBE_LLM_PROVIDER"),
    model: str = typer.Option(None, help="Model name override"),
    top_k: int = typer.Option(3, help="Retrieval candidates per item"),
    data_dir: str = typer.Option(None, envvar="FOODSCRIBE_DATA_DIR"),
) -> None:
    """Process all CSV files in input_dir and save nutrient results to output_dir.

    Each input CSV must have a meal text column (default: 'meal').
    Produces two output files per input CSV:
      <name>_summary.csv  — one row per meal with total macros
      <name>_detail.csv   — one row per ingredient with full nutrient data
    """
    import pandas as pd

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    if not in_dir.exists():
        typer.echo(f"[error] Input folder not found: {in_dir}", err=True)
        raise typer.Exit(1)

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        typer.echo(f"[error] No CSV files found in {in_dir}", err=True)
        raise typer.Exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    ddir = Path(data_dir) if data_dir else DATA_DIR
    llm, retriever, nut_lookup, analyser = _make_pipeline(provider, model, top_k, ddir)
    typer.echo(f"Provider: {llm.provider} ({llm.model})")
    typer.echo(f"Processing {len(csv_files)} file(s) from {in_dir}/ -> {out_dir}/\n")

    def _run_meal(text: str):
        items = llm.parse_meal(text)
        if not items:
            return []
        queries = [i.item for i in items]
        batch_results = retriever.retrieve_batch(queries, top_k=top_k, contexts=[text] * len(queries))
        rows = []
        for item, cands in zip(items, batch_results):
            if not cands:
                continue
            row = nut_lookup.get_scaled(cands[0].fdc_id, item.grams) if item.grams else nut_lookup.get(cands[0].fdc_id)
            if row:
                row._confidence = item.confidence
                row._grams = item.grams
                rows.append(row)
        return rows

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if meal_col not in df.columns:
            typer.echo(f"[skip] {csv_path.name} — no '{meal_col}' column", err=True)
            continue

        typer.echo(f"[{csv_path.name}] {len(df)} rows")
        summary_records = []
        detail_records = []

        for idx, meal_text in enumerate(df[meal_col], start=1):
            meal_text = str(meal_text).strip()
            if not meal_text:
                continue
            try:
                rows = _run_meal(meal_text)
            except Exception as exc:
                typer.echo(f"  row {idx}: [error] {exc}", err=True)
                continue

            if not rows:
                typer.echo(f"  row {idx}: [skip] no food items recognised")
                continue

            summary = analyser.summarise(rows)
            summary_records.append({
                "row": idx,
                "meal": meal_text,
                "energy_kcal": round(summary.total_energy_kcal, 1),
                "protein_g":   round(summary.total_protein_g, 1),
                "carb_g":      round(summary.total_carb_g, 1),
                "fat_g":       round(summary.total_fat_g, 1),
                "fiber_g":     round(summary.total_fiber_g, 1),
                "item_count":  summary.item_count,
            })

            from foodscribe.analysis.stats import _f
            for r in rows:
                detail_records.append({
                    "row":        idx,
                    "meal":       meal_text,
                    "ingredient": r.description,
                    "grams":      getattr(r, "_grams", None) or 100,
                    "energy_kcal": round(_f(r.energy_kcal), 1),
                    "protein_g":   round(_f(r.protein_g), 1),
                    "carb_g":      round(_f(r.carb_g), 1),
                    "fat_g":       round(_f(r.fat_g), 1),
                    "fiber_g":     round(_f(r.fiber_g), 1),
                    "category":    r.category if hasattr(r, "category") else "",
                    "subcategory": r.subcategory if hasattr(r, "subcategory") else "",
                    "confidence":  getattr(r, "_confidence", None),
                })

            typer.echo(f"  row {idx}: {len(rows)} ingredient(s), {summary.total_energy_kcal:.0f} kcal")

        stem = csv_path.stem
        if summary_records:
            summary_path = out_dir / f"{stem}_summary.csv"
            pd.DataFrame(summary_records).to_csv(summary_path, index=False)
            typer.echo(f"  -> {summary_path}")
        if detail_records:
            detail_path = out_dir / f"{stem}_detail.csv"
            pd.DataFrame(detail_records).to_csv(detail_path, index=False)
            typer.echo(f"  -> {detail_path}")

    typer.echo("\nBatch complete.")


if __name__ == "__main__":
    app()
