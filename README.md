# FoodScribe

LLM-powered food diary parser with USDA nutrient lookup.

FoodScribe turns free-text meal descriptions into structured nutrient profiles grounded in the USDA FoodData Central database. It uses a four-stage pipeline:

1. **LLM parsing** — breaks a meal description into individual ingredients with estimated quantities
2. **Semantic retrieval** — matches each ingredient to a USDA food entry using MPNet embeddings
3. **Nutrient lookup** — pulls exact USDA nutrient values (macros + micros) for the matched food
4. **Stats & visualisation** — aggregates nutrients per meal, per day, or across a batch

---

## Installation

```bash
git clone 
cd foodscribe
pip install -e .
```

### Requirements
- Python 3.11+
- An API key for at least one supported LLM provider (see [Environment variables](#environment-variables))

---

## Data Setup

FoodScribe requires pre-built lookup tables derived from the USDA FoodData Central CSVs. Download the long-format CSVs (Foundation, SR Legacy, Survey/FNDDS) and run:

```bash
python scripts/build_data.py --usda-dir /path/to/USDA_data/ --data-dir data/
```

This produces four files in `data/`:
- `food_metadata.csv` — food descriptions and categories
- `food_categories.csv` — category/subcategory per food
- `foods_wide.csv` — all nutrients per food (wide format, units in column names)
- `food_embeddings_mpnet.npy` — 768-dim MPNet embeddings for semantic search

Add `--skip-embeddings` to skip the slow embedding step (requires GPU/CPU for ~30 min).

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
foodscribe parse-meal "2 scrambled eggs with toast and orange juice"
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

### Retrieve nutrients for a specific USDA food
```bash
foodscribe lookup-nutrients --fdc-id 171705
foodscribe categories --fdc-id 171705
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

Batch runs produce timestamped folders under `output/`:

```
output/
└── run_20260311_173822/
    ├── journal_parsed.csv    # LLM output: one row per ingredient
    ├── journal_detail.csv    # Nutrient detail: full USDA data per ingredient
    └── journal_summary.csv   # Per-meal totals: all macro + micro nutrients
```

---

## License

MIT — see [LICENSE](LICENSE).
