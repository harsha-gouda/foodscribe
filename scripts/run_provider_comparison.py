"""
Run batch-parse + batch-nutrients for three LLM providers on the 2-week diet data.

Usage:
    python scripts/run_provider_comparison.py
    python scripts/run_provider_comparison.py --limit 500 --input input/parsed_diet_2weeks.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable

PROVIDERS = [
    ("anthropic", "claude-sonnet-4-6"),
    ("openai",    None),   # gpt-4o-mini
    ("deepseek",  None),   # deepseek-chat
]


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="input/parsed_diet_2weeks.csv")
    parser.add_argument("--meal-col", default="meal")
    parser.add_argument("--limit",    type=int, default=500,
                        help="Max rows per provider (0 = all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)

    run_ids: dict[str, str] = {}

    for provider, model in PROVIDERS:
        # ── Step 1: batch-parse ──────────────────────────────────────────
        parse_cmd = [
            PYTHON, "-c", "from foodscribe.cli import app; app()",
            "--", "batch-parse", str(input_path),
            "--meal-col", args.meal_col,
            "--provider", provider,
            "--limit", str(args.limit),
        ]
        if model:
            parse_cmd += ["--model", model]

        rc = run(parse_cmd, f"[{provider}] batch-parse  (limit={args.limit})")
        if rc != 0:
            print(f"[warn] batch-parse failed for {provider} (exit {rc}) — skipping nutrients step")
            continue

        # ── Step 2: batch-nutrients (auto-detects latest run folder) ─────
        nutrients_cmd = [
            PYTHON, "-c", "from foodscribe.cli import app; app()",
            "--", "batch-nutrients",
        ]
        run(nutrients_cmd, f"[{provider}] batch-nutrients")

    print("\n\nDone. Check output/ for timestamped run folders (one per provider).")
    print("Tip: ls output/ | sort  — the three most recent folders are the comparison runs.")


if __name__ == "__main__":
    main()
