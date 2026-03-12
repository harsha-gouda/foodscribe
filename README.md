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
git clone https://github.com/harsha-gouda/foodscribe.git
cd foodscribe
pip install -e .
```

### Requirements
- Python 3.11+
- An API key for at least one supported LLM provider (see [Environment variables](#environment-variables))

---

## Data Setup

### Option 1 — Download pre-built files (recommended)

Download the four pre-built data files and place them in the `data/` folder:

| File | Description |
|------|-------------|
| `food_metadata.csv` | Food descriptions and categories |
| `food_categories.csv` | Category/subcategory per food |
| `foods_wide.csv` | All nutrients per food (wide format) |
| `food_embeddings_mpnet.npy` | 768-dim MPNet embeddings for semantic search |

> **Download:** [10.5281/zenodo.18990541.](https://doi.org/10.5281/zenodo.18990542)

```bash
# After downloading, move all four files into:
data/
```

### Option 2 — Build from raw USDA CSVs (advanced)

If you want to rebuild the data files yourself, download the long-format CSVs from [USDA FoodData Central](https://fdc.nal.usda.gov/download-datasets) (Foundation, SR Legacy, Survey/FNDDS) and run:

```bash
python scripts/build_data.py --usda-dir /path/to/USDA_data/ --data-dir data/
# Add --skip-embeddings to skip the slow embedding step (~30 min on CPU)
```

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
