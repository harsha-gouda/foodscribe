# FoodScribe

LLM-powered food diary parser with USDA nutrient lookup.

FoodScribe turns free-text meal descriptions into structured nutrient profiles grounded in the USDA FoodData Central database. It uses a four-stage pipeline:

1. **LLM parsing** — breaks a meal description into individual ingredients with estimated quantities
2. **Semantic retrieval** — matches each ingredient to a USDA food entry using OpenAI `text-embedding-3-large` embeddings (falls back to MPNet if OpenAI index is not present)
3. **Nutrient lookup** — pulls exact USDA nutrient values (macros + micros) for the matched food, scaled to gram weight
4. **Stats & visualisation** — aggregates nutrients per meal, per day, or across a batch

---

## Installation

```bash
git clone https://github.com/harsha-gouda/foodscribe.git
cd foodscribe
pip install -e .
```

### Requirements
- Python 3.11+
- An API key for at least one supported LLM provider (see [Environment variables](#environment-variables))

---

## Data Setup

Download the four pre-built data files and place them in the `data/` folder:

| File | Description |
|------|-------------|
| `food_metadata.csv` | Food descriptions and categories |
| `food_categories.csv` | Category/subcategory per food |
| `foods_wide.csv` | All nutrients per food (wide format) |
| `food_embeddings_mpnet.npy` | 768-dim MPNet embeddings for semantic search |

> **Download:** [10.5281/zenodo.18990542](https://doi.org/10.5281/zenodo.18990542)

```bash
# After downloading, move all four files into:
data/
```

### Optional: OpenAI embedding index (higher accuracy)

For better semantic matching accuracy, build a `text-embedding-3-large` index (3072-dim). Requires an OpenAI API key and the raw USDA source data:

```bash
python scripts/build_data.py --usda-dir ../USDA_data/ --embedder openai
```

This writes `data/food_embeddings_openai.npy`. When present, FoodScribe automatically uses it instead of the MPNet index.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `FOODSCRIBE_LLM_PROVIDER` | Default provider (`anthropic`, `openai`, `deepseek`, `gemini`) |
| `FOODSCRIBE_LLM_MODEL` | Default model (optional, overrides provider default) |
| `FOODSCRIBE_DATA_DIR` | Path to `data/` folder (default: `./data`) |

Set a default provider so you don't need `--provider` on every command:

```bash
foodscribe use-provider anthropic
```

---

## Quick Start

### Parse a single meal

```bash
foodscribe parse "2 scrambled eggs with toast and orange juice"
```

```
Identified: scrambled eggs (100g) · bread, toasted (28g) · orange juice (248g)

                                           Meal Nutrient Profile                                           
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Food (USDA match)                     ┃ Grams ┃ Energy(kcal) ┃ Protein(g) ┃ Carb(g) ┃ Fat(g) ┃ Fiber(g) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Eggs, scrambled, frozen mixture       │   100 │          131 │       13.1 │     7.5 │    5.6 │      0.0 │
│ Bread, wheat, toasted                 │    28 │           88 │        3.6 │    15.6 │    1.2 │      1.3 │
│ Orange juice, 100%,  freshly squeezed │   248 │          117 │        2.0 │    24.8 │    0.9 │      0.7 │
├───────────────────────────────────────┼───────┼──────────────┼────────────┼─────────┼────────┼──────────┤
│ TOTAL                                 │       │          335 │       18.7 │    47.9 │    7.7 │      2.1 │
└───────────────────────────────────────┴───────┴──────────────┴────────────┴─────────┴────────┴──────────┘

Macronutrient split:  Protein 22%  |  Carbs 57%  |  Fat 21%
Categories present:   Baked Products·Dairy and Egg Products
Dominant category:    Dairy and Egg Products  (131 kcal, 39% of meal energy)
```

With full micronutrient panel:

```bash
foodscribe parse "2 scrambled eggs with toast" --all-nutrients
```

### Batch process a food journal CSV

```bash
# Step 1 — LLM parsing (CSV must have a column with meal descriptions)
foodscribe batch-parse input/journal.csv --meal-col meal --limit 200

# Step 2 — Nutrient lookup
foodscribe batch-nutrients

# Resume an interrupted run
foodscribe batch-parse input/journal.csv --run-id run_20260311_173822 --limit 200
```

### Direct ingredient lookup (no LLM)

If you already have a CSV with explicit ingredient names and gram weights, skip the LLM step entirely:

```bash
foodscribe ingredient-lookup input/food_ingredients.csv
```

The `Ingredient` and `grams` columns are used directly for semantic retrieval and nutrient scaling. All extra columns (e.g. `ParticipantCode`, `RecordID`) are carried through to the output.

```bash
# Custom column names
foodscribe ingredient-lookup input/foods.csv --ingredient-col FoodName --grams-col WeightG

# Or use the standalone script
python scripts/ingredient_lookup.py input/food_ingredients.csv
```

### Aggregate to daily totals

```bash
foodscribe aggregate output/run_xyz/meals_summary.csv "SubjectID,Date" --no-meal
```

### Look up a specific USDA food

```bash
foodscribe categories --fdc-id 171705
foodscribe categories --list
foodscribe categories --filter "Poultry Products"
```

---

## Supported LLM Providers

| Provider | Default Model |
|----------|---------------|
| `anthropic` | claude-haiku-4-5-20251001 |
| `openai` | gpt-4o-mini |
| `deepseek` | deepseek-chat |
| `gemini` | gemini-1.5-flash |

---

## Output Files

### Batch runs (`batch-parse` + `batch-nutrients`)

Produces timestamped folders under `output/`:

```
output/
└── run_20260311_173822/
    ├── journal_parsed.csv      # LLM output: one row per ingredient
    ├── journal_summary.csv     # Per-meal totals: all macro + micro nutrients
    ├── journal_detail.csv      # Per-ingredient USDA match + scaled nutrients
    ├── journal_food_items.csv  # Pivot: rows=meals, columns=USDA food items (grams)
    └── journal_food_groups.csv # Pivot: rows=meals, columns=food categories (grams)
```

### ingredient-lookup

```
input/
└── food_ingredients_nutrients.csv  # One row per ingredient: USDA match + scaled nutrients
```

### Summary columns (`*_summary.csv`)

| Column | Description |
|--------|-------------|
| `meal` | Original meal text |
| `ingredients` | Semicolon-separated ingredient list |
| `avg_confidence` | Mean LLM confidence score (1–5) |
| `item_count` | Number of matched USDA ingredients |
| `dominant_category` | USDA category contributing most energy |
| `energy_kcal`, `protein_g`, … | All nutrients summed across ingredients |

### Detail columns (`*_detail.csv`)

| Column | Description |
|--------|-------------|
| `ingredient` | Parsed ingredient name |
| `fdc_id` | USDA FoodData Central ID |
| `usda_match` | Matched USDA food description |
| `grams` | Gram weight used for scaling |
| `category` / `subcategory` | USDA food category |
| `data_type` | Foundation, SR Legacy, or Survey |
| `confidence` | LLM confidence score (1–5) |
| all nutrient columns | Scaled to gram weight |

---

## Typical Workflows

### Large batch with resume support

```bash
# First run — auto-creates output/run_YYYYMMDD_HHMMSS/
foodscribe batch-parse input/journal.csv --limit 200

# Resume (same run folder, skips already-processed rows)
foodscribe batch-parse input/journal.csv --run-id run_20260311_173822 --limit 200

# Nutrient retrieval — auto-detects latest run folder
foodscribe batch-nutrients

# Roll up to daily totals per subject
foodscribe aggregate output/run_20260311_173822/journal_summary.csv "SubjectID,Date" --no-meal
```

### Re-run retrieval only (e.g. after rebuilding the embedding index)

```bash
foodscribe batch-nutrients --run-id run_20260311_173822
```

---

## Python API

```python
from foodscribe.analysis.stats import MealAnalyser, MealSummary

analyser = MealAnalyser()
summary = analyser.summarise(rows)       # MealSummary with macro totals
analyser.print_table(rows, summary)      # Rich terminal table
analyser.plot_macros_pie(summary)        # Pie chart of macro split
analyser.plot_energy_distribution(rows)  # Energy per food item
analyser.plot_nutrient_bars(rows)        # Protein/carb/fat per item
analyser.plot_category_breakdown(rows)   # Energy by USDA category
```

```python
import pandas as pd

summary    = pd.read_csv("output/run_xyz/meals_summary.csv")
detail     = pd.read_csv("output/run_xyz/meals_detail.csv")
food_items = pd.read_csv("output/run_xyz/meals_food_items.csv")

# Average daily calories per subject
summary.groupby("SubjectID")["energy_kcal"].mean()

# All dairy ingredients
detail[detail["category"] == "Dairy and Egg Products"]

# Meals above 800 kcal
summary[summary["energy_kcal"] > 800][["SubjectID", "Date", "meal", "energy_kcal"]]
```
