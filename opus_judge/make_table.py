#!/usr/bin/env python3
"""Build the results table from judged records.

Per pattern, the mean is taken over every score of the metric, zeros included:
27 Q1 passage scores, 18 Q2 scores, 9 Q3-M scores and 9 Q3-T scores. Avg is the
unweighted mean of the P1, P2 and P3 means. This is the aggregation behind
results/final/results_table.csv in the released repository.

  python3 -I opus_judge/make_table.py                # print the table
  python3 -I opus_judge/make_table.py --write        # also write CSV, LaTeX rows, summaries

With --write, and when the released GPT-6.1 Terra records are available as the
reference (results/final in the repository), it also writes the mean of the two
judges (results_table_mean.csv) and the paper's three-panel table
(table_three_panels.tex): the mean, then each judge separately. Both judges
scored the same items, so the mean of the two tables equals the table of
per-item means.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next((c for c in (HERE.parent, HERE.parent / "repo_latest") if (c / "experiments" / "sampling.py").exists()), HERE.parent)

MODELS = [  # directory, name in results_table.csv, name in the paper's table
    ("gpt-4.1", "GPT-4.1", "GPT-4.1*"),
    ("gpt-oss-high", "GPT-OSS-120B High", "OSS-High"),
    ("gpt-oss-medium", "GPT-OSS-120B Medium", "OSS-Medium"),
    ("gpt-5.6-luna", "GPT-5.6-Luna", "GPT-5.6-Luna*"),
]
METRICS = {  # column block: (experiment, metric key, scores per pattern)
    "Q1": ("1", "E_text_alignment_scores", 27),
    "Q2": ("2", "E_text_alignment_score", 18),
    "Q3_M": ("3", "M_alignment_score", 9),
    "Q3_T": ("3", "T_alignment_score", 9),
}
PATTERNS = ["P1", "P2", "P3"]
CSV_BLOCKS = ["Q1", "Q2", "Q3_T", "Q3_M"]    # column order of the repository's CSV
PAPER_BLOCKS = ["Q1", "Q2", "Q3_M", "Q3_T"]  # column order of the paper's table
FILES = ["p1-p2-evaluations.jsonl", "p3-evaluations.jsonl"]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect(evals: Path, allow_partial: bool = False):
    """{directory: {(block, pattern): Fraction mean}} plus the set of judge models seen."""
    table: dict[str, dict[tuple[str, str], Fraction]] = {}
    judges: set[str] = set()
    for directory, _, _ in MODELS:
        scores: dict[tuple[str, str], list[int]] = {(b, p): [] for b in METRICS for p in PATTERNS}
        for name in FILES:
            path = evals / directory / name
            if not path.exists():
                sys.exit(f"missing {path}")
            for record in load(path):
                if record.get("errors"):
                    sys.exit(f"record with errors in {path}: {record['task_id']}")
                judges.add(str(record.get("judge_model")))
                for block, (experiment, metric, _) in METRICS.items():
                    if str(record["experiment"]) == experiment and metric in record["metrics"]:
                        value = record["metrics"][metric]
                        scores[(block, record["pattern_id"])].extend(value if isinstance(value, list) else [value])
        means = {}
        for (block, pattern), values in scores.items():
            expected = METRICS[block][2]
            if len(values) != expected and not allow_partial:
                sys.exit(f"{directory} {block} {pattern}: {len(values)} scores, expected {expected}")
            means[(block, pattern)] = Fraction(sum(values), len(values)) if values else None
        for block in METRICS:
            parts = [means[(block, p)] for p in PATTERNS]
            means[(block, "Avg")] = sum(parts, Fraction(0)) / len(parts) if None not in parts else None
        table[directory] = means
    return table, judges


def csv_text(table) -> str:
    header = ["model"] + [f"{b}_{c}" for b in CSV_BLOCKS for c in PATTERNS + ["Avg"]]
    lines = [",".join(header)]
    for directory, csv_name, _ in sorted(MODELS, key=lambda m: m[1]):
        cells = [repr(round(float(table[directory][(b, c)]), 4)) for b in CSV_BLOCKS for c in PATTERNS + ["Avg"]]
        lines.append(",".join([csv_name] + cells))
    return "\n".join(lines) + "\n"


def two(value: Fraction) -> str:
    """Two decimals, rounded half up from the exact value."""
    exact = Decimal(value.numerator) / Decimal(value.denominator)
    return str(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


TABLE_HEAD = r"""\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrrrrrrrrrrrrrrr}
\toprule
& \multicolumn{4}{c}{Q1: Example mining}
& \multicolumn{4}{c}{Q2: Example alignment}
& \multicolumn{4}{c}{Q3: $M$}
& \multicolumn{4}{c}{Q3: $T$} \\
\cmidrule(lr){2-5}
\cmidrule(lr){6-9}
\cmidrule(lr){10-13}
\cmidrule(lr){14-17}
Model
& P1 & P2 & P3 & Avg.
& P1 & P2 & P3 & Avg.
& P1 & P2 & P3 & Avg.
& P1 & P2 & P3 & Avg. \\
\midrule
"""
CAPTION = ("Executor results by pattern and query: (a) the mean of two judges, GPT-6.1 Terra and Claude Opus 5, "
           "both with reasoning disabled; (b) and (c) each judge separately. Each score is the mean over three "
           "sampled executions per task. Avg.\\ is the unweighted mean across P1, P2, and P3. Asterisks denote "
           "models run without reasoning.")
PANELS = ["Mean of the two judges", "Judge: GPT-6.1 Terra", "Judge: Claude Opus 5"]


def panel_cells(table) -> list[tuple[str, list[str]]]:
    """[(model name, 16 cells)], the best Avg of each block in bold."""
    best = {b: max(table[d][(b, "Avg")] for d, _, _ in MODELS) for b in PAPER_BLOCKS}
    out = []
    for directory, _, paper_name in MODELS:
        cells = []
        for block in PAPER_BLOCKS:
            cells += [two(table[directory][(block, p)]) for p in PATTERNS]
            avg = two(table[directory][(block, "Avg")])
            cells.append(f"\\textbf{{{avg}}}" if table[directory][(block, "Avg")] == best[block] else avg)
        out.append((paper_name, cells))
    return out


def three_panel_table(mean, reference, mine, wrap=None, caption=None) -> str:
    """The paper's results table. wrap(panel index, text) lets a marked build colour what is new."""
    wrap = wrap or (lambda index, text: text)
    blocks = []
    for index, (title, table) in enumerate(zip(PANELS, (mean, reference, mine))):
        label = wrap(None, f"\\emph{{({chr(97 + index)}) {title}}}")
        lines = [f"\\multicolumn{{17}}{{@{{}}l}}{{{label}}} \\\\"]
        for name, cells in panel_cells(table):
            lines.append(wrap(index, name) + " & " + " & ".join(wrap(index, cell) for cell in cells) + " \\\\")
        blocks.append("\n".join(lines))
    return ("\\begin{table*}[t]\n\\centering\n" + (caption or f"\\caption{{{CAPTION}}}") + "\n\\label{tab:robustness-all}\n"
            + TABLE_HEAD + "\n\\midrule\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def mean_of(a, b):
    return {d: {k: (a[d][k] + b[d][k]) / 2 for k in a[d]} for d in a}


def tex_rows(table) -> str:
    best = {b: max(table[d][(b, "Avg")] for d, _, _ in MODELS) for b in PAPER_BLOCKS}
    rows = []
    for directory, _, paper_name in MODELS:
        lines = [paper_name]
        for index, block in enumerate(PAPER_BLOCKS):
            cells = [two(table[directory][(block, p)]) for p in PATTERNS]
            avg = two(table[directory][(block, "Avg")])
            cells.append(f"\\textbf{{{avg}}}" if table[directory][(block, "Avg")] == best[block] else avg)
            lines.append("& " + " & ".join(cells) + (" \\\\" if index == len(PAPER_BLOCKS) - 1 else ""))
        rows.append("\n".join(lines))
    return "\n\n".join(rows) + "\n"


def plain(table) -> str:
    out = [f"{'':14s}" + "".join(f"{b + ' (P1 P2 P3 Avg)':>28s}" for b in PAPER_BLOCKS)]
    for directory, _, paper_name in MODELS:
        out.append(f"{paper_name:14s}" + "".join(
            "    " + " ".join(two(table[directory][(b, c)]) for c in PATTERNS + ["Avg"]) for b in PAPER_BLOCKS))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evals", type=Path, default=HERE / "results" / "claude-opus-5")
    parser.add_argument("--reference", type=Path, default=REPO / "results" / "final",
                        help="records of the other judge, for the mean of the two and the three-panel table")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    evals = args.evals.resolve()
    table, judges = collect(evals)
    print(f"judge: {sorted(judges)}\n")
    print(plain(table))
    if args.write:
        (evals / "results_table.csv").write_text(csv_text(table), encoding="utf-8")
        (evals / "table_rows.tex").write_text(tex_rows(table), encoding="utf-8")
        sys.path.append(str(REPO))
        from experiments.sampling import summarize_samples

        for directory, _, _ in MODELS:
            for name in FILES:
                summary = summarize_samples(evals / directory / name)
                target = evals / directory / name.replace("evaluations.jsonl", "summary.json")
                target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {evals / 'results_table.csv'}, {evals / 'table_rows.tex'} and per-model summaries")
        reference_dir = args.reference.resolve()
        if reference_dir != evals and all((reference_dir / d / name).exists() for d, _, _ in MODELS for name in FILES):
            reference, _ = collect(reference_dir)
            mean = mean_of(table, reference)
            (evals / "results_table_mean.csv").write_text(csv_text(mean), encoding="utf-8")
            (evals / "table_three_panels.tex").write_text(three_panel_table(mean, reference, table), encoding="utf-8")
            print("\nmean of the two judges\n" + plain(mean))
            print("wrote results_table_mean.csv and table_three_panels.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
