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

Download pre-built embedding files

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
Example1: foodscribe parse "2 scrambled eggs with toast and orange juice"

Output: Meal Nutrient Profile                                           
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Food (USDA match)                     ┃ Grams ┃ Energy(kcal) ┃ Protein(g) ┃ Carb(g) ┃ Fat(g) ┃ Fiber(g) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Egg, whole, cooked, scrambled         │   100 │          149 │       10.0 │     1.6 │   11.0 │      0.0 │
│ Bread, white, toasted                 │    56 │          164 │        5.8 │    30.3 │    2.2 │      1.4 │
│ Orange juice, 100%,  freshly squeezed │   240 │          113 │        1.9 │    24.0 │    0.9 │      0.7 │
├───────────────────────────────────────┼───────┼──────────────┼────────────┼─────────┼────────┼──────────┤
│ TOTAL                                 │       │          426 │       17.8 │    55.9 │   14.1 │      2.1 │
└───────────────────────────────────────┴───────┴──────────────┴────────────┴─────────┴────────┴──────────┘

Macronutrient split:  Protein 17%  |  Carbs 53%  |  Fat 30%

Example 2: foodscribe parse "for dinner i had 200g of chicken biriyani"

Output: Meal Nutrient Profile                                             
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Food (USDA match)                        ┃ Grams ┃ Energy(kcal) ┃ Protein(g) ┃ Carb(g) ┃ Fat(g) ┃ Fiber(g) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Chicken, ground, raw                     │   150 │          214 │       26.1 │     0.1 │   12.1 │      0.0 │
│ Rice, white, medium-grain, cooked, unenr │    50 │           65 │        1.2 │    14.3 │    0.1 │      0.0 │
│ Onions, raw                              │    30 │           11 │        0.3 │     2.5 │    0.0 │      0.5 │
│ Yogurt, plain, whole milk                │    20 │            0 │        0.8 │     1.1 │    0.9 │      0.0 │
│ Oil, vegetable, soybean, refined         │     8 │           71 │        0.0 │     0.0 │    8.0 │      0.0 │
│ Spices, garlic powder                    │     5 │           17 │        0.8 │     3.6 │    0.0 │      0.5 │
│ Spices, cumin seed                       │     1 │            4 │        0.2 │     0.4 │    0.2 │      0.1 │
│ Spices, cinnamon, ground                 │     0 │            1 │        0.0 │     0.4 │    0.0 │      0.3 │
│ Spices, cardamom                         │     0 │            2 │        0.1 │     0.3 │    0.0 │      0.1 │
│ Salt, table                              │     2 │            0 │        0.0 │     0.0 │    0.0 │      0.0 │
├──────────────────────────────────────────┼───────┼──────────────┼────────────┼─────────┼────────┼──────────┤
│ TOTAL                                    │       │          385 │       29.4 │    22.8 │   21.5 │      1.5 │
└──────────────────────────────────────────┴───────┴──────────────┴────────────┴─────────┴────────┴──────────┘

Macronutrient split:  Protein 29%  |  Carbs 23%  |  Fat 48%

Example 3: foodscribe parse "for lunch i has a small plate of fish and chips with a can of coca cola"

Output: Meal Nutrient Profile                                            
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Food (USDA match)                       ┃ Grams ┃ Energy(kcal) ┃ Protein(g) ┃ Carb(g) ┃ Fat(g) ┃ Fiber(g) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Fish, mackerel, fried                   │   150 │          426 │       25.8 │    17.5 │   27.3 │      0.8 │
│ Potato, french fries, from fresh, fried │   120 │          238 │        2.3 │    22.2 │   15.7 │      1.9 │
│ Soft drink, cola                        │   375 │          158 │        0.0 │    39.0 │    0.9 │      0.0 │
├─────────────────────────────────────────┼───────┼──────────────┼────────────┼─────────┼────────┼──────────┤
│ TOTAL                                   │       │          821 │       28.1 │    78.8 │   44.0 │      2.7 │
└─────────────────────────────────────────┴───────┴──────────────┴────────────┴─────────┴────────┴──────────┘

Macronutrient split:  Protein 14%  |  Carbs 38%  |  Fat 48%

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
