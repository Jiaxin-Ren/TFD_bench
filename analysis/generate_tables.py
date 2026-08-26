"""
Table Generator — generate publication-ready tables from collected results.

Usage:
    python analysis/generate_tables.py --input results/summary.json
    python analysis/generate_tables.py --format latex --highlight best
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.methods import (
    BENCHMARK_METHODS,
    METHOD_ABBREVIATIONS,
    method_sort_key,
)  # noqa: E402

DEFAULT_METRICS = ("test/cls/Acc", "test/cal/ECE", "ood/AUROC")

METRIC_CONFIG = {
    "test/cls/Acc":     {"name": "ACC",          "higher_better": True,  "format": ".2f", "scale": 100},
    "test/cls/NLL":     {"name": "NLL",           "higher_better": False, "format": ".4f", "scale": 1},
    "test/cls/Brier":   {"name": "Brier",         "higher_better": False, "format": ".4f", "scale": 1},
    "test/cls/Entropy": {"name": "Entropy",       "higher_better": False, "format": ".4f", "scale": 1},
    "test/cal/ECE":     {"name": "ECE",           "higher_better": False, "format": ".2f", "scale": 100},
    "test/cal/aECE":    {"name": "aECE",          "higher_better": False, "format": ".4f", "scale": 1},
    "ood/AUROC":        {"name": "AUROC",         "higher_better": True,  "format": ".2f", "scale": 100},
    "ood/overall_AUROC":{"name": "Overall AUROC", "higher_better": True,  "format": ".2f", "scale": 100},
    "ood/macro_AUROC":  {"name": "Macro AUROC",   "higher_better": True,  "format": ".2f", "scale": 100},
    "ood/AUPR":         {"name": "AUPR",          "higher_better": True,  "format": ".2f", "scale": 100},
    "ood/FPR95":        {"name": "FPR95",         "higher_better": False, "format": ".2f", "scale": 100},
    "test/sc/AURC":     {"name": "AURC",          "higher_better": False, "format": ".4f", "scale": 1},
    "test/sc/AUGRC":    {"name": "AUGRC",         "higher_better": False, "format": ".4f", "scale": 1},
    "test/sc/Cov@5Risk":{"name": "Cov@5%Risk",   "higher_better": True,  "format": ".2f", "scale": 100},
    "test/sc/Risk@80Cov":{"name":"Risk@80%Cov",  "higher_better": False, "format": ".4f", "scale": 1},
}


def _metric_config(metric: str) -> dict[str, Any]:
    if metric.startswith("ood/source_AUROC/"):
        source = metric.removeprefix("ood/source_AUROC/")
        return {
            "name": f"AUROC ({source})",
            "higher_better": True,
            "format": ".2f",
            "scale": 100,
        }
    return METRIC_CONFIG.get(metric, {"name": metric})

METHOD_NAMES = METHOD_ABBREVIATIONS
CONFIG_NAMES = {
    "clean": "Clean",
    "operating_shift": "Operating shift",
}


def load_results(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(mean, std, config, highlight, fmt_type):
    scale = config.get("scale", 1)
    f = config.get("format", ".4f")
    m, s = mean * scale, std * scale
    val = f"{m:{f}}" if s == 0 else f"{m:{f}}±{s:{f}}"
    if highlight:
        if fmt_type == "markdown": return f"**{val}**"
        if fmt_type == "latex":    return f"\\textbf{{{val}}}"
        if fmt_type == "html":     return f"<b>{val}</b>"
    return val


def _best(results, metrics):
    best = {}
    for metric in metrics:
        higher = _metric_config(metric).get("higher_better", True)
        best_key, best_val = None, None
        for key, stats in results.items():
            ms = stats.get("metrics", {}).get(metric)
            if ms is None:
                continue
            v = ms["mean"]
            if best_val is None or (higher and v > best_val) or (not higher and v < best_val):
                best_val, best_key = v, key
        if best_key:
            best[metric] = best_key
    return best


def _config_sort_key(config: str):
    """Keep clean and operating shift first, then sort noise severities."""
    if config == "clean":
        return (0, "", 0)
    if config == "operating_shift":
        return (1, "", 0)
    match = re.fullmatch(r"(.+)_s(\d+)", config)
    if match:
        return (2, match.group(1), int(match.group(2)))
    return (3, config, 0)


def _group_results(results):
    """Group methods into separate tables per dataset/backbone/test config."""
    groups = defaultdict(dict)
    for key, stats in results.items():
        method = stats.get("method", "unknown")
        if method not in BENCHMARK_METHODS:
            continue
        group_key = (
            stats.get("dataset", "unknown"),
            stats.get("backbone", "unknown"),
            stats.get("config", "clean"),
        )
        groups[group_key][key] = stats
    for group_key, group in groups.items():
        groups[group_key] = dict(sorted(
            group.items(),
            key=lambda item: method_sort_key(item[1].get("method", "unknown")),
        ))
    return sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], _config_sort_key(item[0][2])),
    )


def _group_title(group_key):
    dataset, backbone, config = group_key
    return f"{dataset} / {backbone} / {CONFIG_NAMES.get(config, config)}"


def generate_markdown_table(results, metrics, highlight_best=True) -> str:
    sections = []
    for group_key, group_results in _group_results(results):
        sections.append(f"## {_group_title(group_key)}")
        sections.append(_generate_markdown_group(group_results, metrics, highlight_best))
    return "\n\n".join(sections)


def _generate_markdown_group(results, metrics, highlight_best=True) -> str:
    best = _best(results, metrics) if highlight_best else {}
    headers = ["Method"] + [
        f"{_metric_config(m)['name']} {'↑' if _metric_config(m).get('higher_better', True) else '↓'}"
        for m in metrics
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for key, stats in results.items():
        method = METHOD_NAMES.get(stats["method"], stats["method"])
        row = [method]
        for metric in metrics:
            ms = stats.get("metrics", {}).get(metric)
            row.append(_fmt(ms["mean"], ms["std"], _metric_config(metric),
                            best.get(metric) == key and highlight_best, "markdown") if ms else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_latex_table(results, metrics, highlight_best=True, caption="") -> str:
    tables = []
    for group_key, group_results in _group_results(results):
        group_caption = caption or _group_title(group_key)
        tables.append(_generate_latex_group(group_results, metrics, highlight_best, group_caption))
    return "\n\n".join(tables)


def _generate_latex_group(results, metrics, highlight_best=True, caption="") -> str:
    best = _best(results, metrics) if highlight_best else {}
    col_spec = "l" + "c" * len(metrics)
    headers = ["Method"] + [
        f"{_metric_config(m)['name']} "
        + ("$\\uparrow$" if _metric_config(m).get("higher_better", True)
           else "$\\downarrow$")
        for m in metrics
    ]
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{{caption}}}" if caption else "", r"\label{tab:results}",
             f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for key, stats in results.items():
        method = METHOD_NAMES.get(stats["method"], stats["method"])
        row = [method]
        for metric in metrics:
            ms = stats.get("metrics", {}).get(metric)
            row.append(_fmt(ms["mean"], ms["std"], _metric_config(metric),
                            best.get(metric) == key and highlight_best, "latex") if ms else "-")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_html_table(results, metrics, highlight_best=True) -> str:
    sections = []
    for group_key, group_results in _group_results(results):
        sections.append(f"<h2>{_group_title(group_key)}</h2>")
        sections.append(_generate_html_group(group_results, metrics, highlight_best))
    return "\n".join(sections)


def _generate_html_group(results, metrics, highlight_best=True) -> str:
    best = _best(results, metrics) if highlight_best else {}
    lines = ['<table border="1" style="border-collapse:collapse"><tr><th>Method</th>']
    for m in metrics:
        cfg = _metric_config(m)
        lines.append(f"<th>{cfg['name']} {'↑' if cfg.get('higher_better', True) else '↓'}</th>")
    lines.append("</tr>")
    for key, stats in results.items():
        method = METHOD_NAMES.get(stats["method"], stats["method"])
        lines.append(f"<tr><td>{method}</td>")
        for metric in metrics:
            ms = stats.get("metrics", {}).get(metric)
            val = _fmt(ms["mean"], ms["std"], _metric_config(metric),
                       best.get(metric) == key and highlight_best, "html") if ms else "-"
            lines.append(f"<td>{val}</td>")
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate tables from results")
    parser.add_argument("--input", default=str(_PROJECT_ROOT / "results" / "summary.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--format", default="markdown", choices=["markdown", "latex", "html"])
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--highlight", default="best", choices=["best", "none"])
    parser.add_argument("--caption", default="")
    parser.add_argument(
        "--print",
        dest="print_table",
        action="store_true",
        help="Also print the generated table to the terminal",
    )
    args = parser.parse_args()

    results = load_results(args.input)

    metrics = args.metrics or list(DEFAULT_METRICS)

    highlight = args.highlight == "best"
    if args.format == "markdown":
        table = generate_markdown_table(results, metrics, highlight)
    elif args.format == "latex":
        table = generate_latex_table(results, metrics, highlight, args.caption)
    else:
        table = generate_html_table(results, metrics, highlight)

    suffix = {"markdown": "md", "latex": "tex", "html": "html"}[args.format]
    output = (
        Path(args.output)
        if args.output
        else _PROJECT_ROOT / "results" / "tables" / f"table.{suffix}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table, encoding="utf-8")
    print(f"Saved: {output}")

    if args.print_table:
        print(table)


if __name__ == "__main__":
    main()
