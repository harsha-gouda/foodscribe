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

Example 2: foodscribe parse "2 scrambled eggs with toast" --all-nutrients
>>  Meal Nutrient Profile                                          
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Food (USDA match)                   ┃ Grams ┃ Energy(kcal) ┃ Protein(g) ┃ Carb(g) ┃ Fat(g) ┃ Fiber(g) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ Egg, whole, raw                     │   100 │          143 │       12.4 │     1.0 │   10.0 │      0.0 │
│ Butter, without salt                │     5 │           36 │        0.0 │     0.0 │    4.1 │      0.0 │
│ Milk, whole                         │    15 │            9 │        0.5 │     0.7 │    0.5 │      0.0 │
│ Salt, table                         │     1 │            0 │        0.0 │     0.0 │    0.0 │      0.0 │
│ Spices, pepper, black               │     0 │            1 │        0.1 │     0.3 │    0.0 │      0.1 │
│ Bread, white, commercially prepared │    60 │          162 │        5.7 │    29.5 │    2.2 │      1.4 │
│ Butter, without salt                │     7 │           50 │        0.1 │     0.0 │    5.7 │      0.0 │
├─────────────────────────────────────┼───────┼──────────────┼────────────┼─────────┼────────┼──────────┤
│ TOTAL                               │       │          401 │       18.7 │    31.5 │   22.3 │      1.5 │
└─────────────────────────────────────┴───────┴──────────────┴────────────┴─────────┴────────┴──────────┘

Macronutrient split:  Protein 19%  |  Carbs 31%  |  Fat 50%
Categories present:   Baked Products·Dairy and Egg Products·Spices and Herbs
Dominant category:    Baked Products  (162 kcal, 40% of meal energy)
             Full Nutrient Panel (meal totals)              
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Nutrient                                      ┃   Amount ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Alanine (g)                                   │  0.00656 │
│ Alcohol, ethyl (g)                            │        0 │
│ Arginine (g)                                  │  0.00526 │
│ Ash (g)                                       │     2.29 │
│ Aspartic acid (g)                             │   0.0147 │
│ Beta-sitosterol (mg)                          │     0.48 │
│ Betaine (mg)                                  │   0.0445 │
│ Caffeine (mg)                                 │        0 │
│ Calcium, Ca (mg)                              │      198 │
│ Campesterol (mg)                              │        0 │
│ Carbohydrate, by difference (g)               │     31.5 │
│ Carbohydrate, by summation (g)                │     26.9 │
│ Carotene, alpha (µg)                          │     0.06 │
│ Carotene, beta (µg)                           │     21.6 │
│ Cholesterol (mg)                              │      439 │
│ Choline, total (mg)                           │      340 │
│ Copper, Cu (mg)                               │   0.0834 │
│ Cryptoxanthin, beta (µg)                      │     13.1 │
│ Cystine (g)                                   │  0.00165 │
│ Energy (kJ)                                   │ 1.04e+03 │
│ Energy (kcal)                                 │      401 │
│ Fatty acids, total monounsaturated (g)        │     6.98 │
│ Fatty acids, total polyunsaturated (g)        │     3.32 │
│ Fatty acids, total saturated (g)              │       10 │
│ Fatty acids, total trans (g)                  │    0.021 │
│ Fatty acids, total trans-dienoic (g)          │   0.0042 │
│ Fatty acids, total trans-monoenoic (g)        │   0.0168 │
│ Fiber, total dietary (g)                      │     1.51 │
│ Fluoride, F (µg)                              │    0.527 │
│ Folate, DFE (µg)                              │     71.4 │
│ Folate, food (µg)                             │     71.4 │
│ Folate, total (µg)                            │     71.4 │
│ Folic acid (µg)                               │        0 │
│ Fructose (g)                                  │     1.37 │
│ Galactose (g)                                 │  0.00075 │
│ Glucose (g)                                   │    0.865 │
│ Glutamic acid (g)                             │   0.0284 │
│ Glycine (g)                                   │  0.00436 │
│ Histidine (g)                                 │  0.00356 │
│ Hydroxyproline (g)                            │        0 │
│ Iron, Fe (mg)                                 │     3.74 │
│ Isoleucine (g)                                │  0.00795 │
│ Lactose (g)                                   │    0.036 │
│ Leucine (g)                                   │    0.015 │
│ Lutein + zeaxanthin (µg)                      │      505 │
│ Lycopene (µg)                                 │      0.1 │
│ Lysine (g)                                    │  0.00926 │
│ MUFA 14:1 (g)                                 │    8e-05 │
│ MUFA 14:1 c (g)                               │   0.0006 │
│ MUFA 15:1 (g)                                 │        0 │
│ MUFA 16:1 (g)                                 │    0.464 │
│ MUFA 16:1 c (g)                               │    0.128 │
│ MUFA 17:1 (g)                                 │   0.0012 │
│ MUFA 17:1 c (g)                               │   0.0012 │
│ MUFA 18:1 (g)                                 │     5.91 │
│ MUFA 18:1 c (g)                               │     2.45 │
│ MUFA 20:1 (g)                                 │   0.0192 │
│ MUFA 20:1 c (g)                               │   0.0066 │
│ MUFA 22:1 (g)                                 │        0 │
│ MUFA 22:1 c (g)                               │        0 │
│ MUFA 24:1 c (g)                               │        0 │
│ Magnesium, Mg (mg)                            │       30 │
│ Maltose (g)                                   │    0.936 │
│ Manganese, Mn (mg)                            │    0.445 │
│ Methionine (g)                                │    0.003 │
│ Niacin (mg)                                   │     2.88 │
│ Nitrogen (g)                                  │    0.906 │
│ PUFA 18:2 (g)                                 │      1.7 │
│ PUFA 18:2 CLAs (g)                            │   0.0332 │
│ PUFA 18:2 c (g)                               │        1 │
│ PUFA 18:2 i (g)                               │   0.0355 │
│ PUFA 18:2 n-6 c,c (g)                         │     1.26 │
│ PUFA 18:3 (g)                                 │    0.155 │
│ PUFA 18:3 c (g)                               │     0.12 │
│ PUFA 18:3 n-3 c,c,c (ALA) (g)                 │    0.159 │
│ PUFA 18:3 n-6 c,c,c (g)                       │        0 │
│ PUFA 18:4 (g)                                 │        0 │
│ PUFA 20:2 c (g)                               │   0.0012 │
│ PUFA 20:2 n-6 c,c (g)                         │   0.0012 │
│ PUFA 20:3 (g)                                 │  0.00076 │
│ PUFA 20:3 c (g)                               │        0 │
│ PUFA 20:3 n-3 (g)                             │        0 │
│ PUFA 20:3 n-6 (g)                             │        0 │
│ PUFA 20:4 (g)                                 │   0.0018 │
│ PUFA 20:4c (g)                                │   0.0012 │
│ PUFA 20:5 n-3 (EPA) (g)                       │  0.00135 │
│ PUFA 20:5c (g)                                │   0.0012 │
│ PUFA 22:2 (g)                                 │        0 │
│ PUFA 22:4 (g)                                 │   0.0024 │
│ PUFA 22:5 c (g)                               │        0 │
│ PUFA 22:5 n-3 (DPA) (g)                       │   0.0003 │
│ PUFA 22:6 c (g)                               │        0 │
│ PUFA 22:6 n-3 (DHA) (g)                       │        0 │
│ Pantothenic acid (mg)                         │    0.349 │
│ Phenylalanine (g)                             │  0.00715 │
│ Phosphorus, P (mg)                            │      271 │
│ Phytosterols (mg)                             │     0.46 │
│ Potassium, K (mg)                             │      234 │
│ Proline (g)                                   │   0.0169 │
│ Protein (g)                                   │     18.7 │
│ Retinol (µg)                                  │      264 │
│ Riboflavin (mg)                               │    0.589 │
│ SFA 10:0 (g)                                  │    0.319 │
│ SFA 11:0 (g)                                  │        0 │
│ SFA 12:0 (g)                                  │    0.334 │
│ SFA 13:0 (g)                                  │        0 │
│ SFA 14:0 (g)                                  │    0.979 │
│ SFA 15:0 (g)                                  │   0.0024 │
│ SFA 16:0 (g)                                  │     5.35 │
│ SFA 17:0 (g)                                  │   0.0702 │
│ SFA 18:0 (g)                                  │     2.21 │
│ SFA 20:0 (g)                                  │   0.0226 │
│ SFA 22:0 (g)                                  │    0.006 │
│ SFA 24:0 (g)                                  │   0.0024 │
│ SFA 4:0 (g)                                   │      0.4 │
│ SFA 6:0 (g)                                   │     0.25 │
│ SFA 8:0 (g)                                   │    0.152 │
│ Selenium, Se (µg)                             │     45.5 │
│ Serine (g)                                    │  0.00757 │
│ Sodium, Na (mg)                               │      810 │
│ Starch (g)                                    │     22.3 │
│ Stigmasterol (mg)                             │        0 │
│ Sucrose (g)                                   │   0.0001 │
│ Sugars, Total (g)                             │      3.2 │
│ TFA 16:1 t (g)                                │   0.0006 │
│ TFA 18:1 t (g)                                │    0.374 │
│ TFA 18:2 t not further defined (g)            │   0.0042 │
│ TFA 22:1 t (g)                                │        0 │
│ Theobromine (mg)                              │        0 │
│ Thiamin (mg)                                  │    0.391 │
│ Threonine (g)                                 │  0.00578 │
│ Tocopherol, beta (mg)                         │        0 │
│ Tocopherol, delta (mg)                        │        0 │
│ Tocopherol, gamma (mg)                        │   0.0328 │
│ Tocotrienol, alpha (mg)                       │  0.00425 │
│ Tocotrienol, beta (mg)                        │        0 │
│ Tocotrienol, delta (mg)                       │        0 │
│ Tocotrienol, gamma (mg)                       │        0 │
│ Total Sugars (g)                              │    0.932 │
│ Total fat (NLEA) (g)                          │     2.07 │
│ Total lipid (fat) (g)                         │     22.3 │
│ Tryptophan (g)                                │  0.00173 │
│ Tyrosine (g)                                  │  0.00734 │
│ Valine (g)                                    │  0.00958 │
│ Vitamin A, IU (IU)                            │      303 │
│ Vitamin A, RAE (µg)                           │      267 │
│ Vitamin B-12 (µg)                             │     1.12 │
│ Vitamin B-12, added (µg)                      │        0 │
│ Vitamin B-6 (mg)                              │    0.129 │
│ Vitamin C, total ascorbic acid (mg)           │        0 │
│ Vitamin D (D2 + D3) (µg)                      │     2.67 │
│ Vitamin D (D2 + D3), International Units (IU) │        0 │
│ Vitamin D2 (ergocalciferol) (µg)              │        0 │
│ Vitamin D3 (cholecalciferol) (µg)             │        0 │
│ Vitamin E (alpha-tocopherol) (mg)             │     1.34 │
│ Vitamin E, added (mg)                         │        0 │
│ Vitamin K (Dihydrophylloquinone) (µg)         │        0 │
│ Vitamin K (phylloquinone) (µg)                │     2.01 │
│ Water (g)                                     │      112 │
│ Zinc, Zn (mg)                                 │     1.85 │
└───────────────────────────────────────────────┴──────────┘
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

