"""Collect results and generate the standard TFD-Bench table and figures.

Usage:
    python analysis/generate_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.collect_results import (  # noqa: E402
    collect_from_results_dir,
    compute_summary_statistics,
    save_results,
)
from analysis.generate_tables import DEFAULT_METRICS, generate_markdown_table  # noqa: E402
from analysis.visualization.plot_all import generate_all_plots  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "results"
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--skip-table", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    records = collect_from_results_dir(str(results_dir))
    if not records:
        print(f"No experiment results found in {results_dir}")
        return 1

    summary = compute_summary_statistics(records)
    summary_path = results_dir / "summary.json"
    save_results(summary, str(summary_path), "json")

    if not args.skip_table:
        table_path = results_dir / "tables" / "table.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(
            generate_markdown_table(summary, list(DEFAULT_METRICS)),
            encoding="utf-8",
        )
        print(f"Saved: {table_path}")

    if not args.skip_plots:
        generate_all_plots(
            summary,
            results_dir / "figures",
            args.dpi,
            results_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
