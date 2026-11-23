#!/usr/bin/env python3
"""Agreement between this folder's judge and the released GPT-6.1 Terra judge.

The two judges scored the same generator outputs with the same prompts. An item
is one score: a Q1 passage, a Q2 passage, or a Q3 M or T. Items that the
grounding rule scores 0 are excluded, because that zero is assigned by code
before any judge is called and both judges therefore share it by construction.

Reported for the 1 to 5 scores: exact and within-one agreement, Cohen's kappa
(unweighted, linear and quadratic weights), Krippendorff's alpha (ordinal),
Spearman's rho, and the mean difference. Intervals are 95% percentile bootstrap
intervals, resampling tasks (all items of a task stay together).

  python3 -I agreement.py            # print
  python3 -I agreement.py --write    # also write agreement_with_gpt-6.1-terra.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
import make_table as mt  # noqa: E402

LEVELS = [1, 2, 3, 4, 5]
COMPONENTS = {
    "E_text_alignment_scores": "Q1",
    "E_text_alignment_score": "Q2",
    "M_alignment_score": "Q3 M",
    "T_alignment_score": "Q3 T",
}


def load_items(evals: Path) -> dict[tuple, dict]:
    """{(model, sample_id, metric, position): {score, task, component, pattern}}"""
    items = {}
    for directory, _, _ in mt.MODELS:
        for name in mt.FILES:
            for record in mt.load(evals / directory / name):
                if record.get("errors"):
                    sys.exit(f"record with errors in {evals / directory / name}")
                for metric, value in record["metrics"].items():
                    for position, score in enumerate(value if isinstance(value, list) else [value]):
                        items[(directory, record["sample_id"], metric, position)] = {
                            "score": score,
                            "task": record["task_id"],
                            "component": COMPONENTS[metric],
                            "pattern": record["pattern_id"],
                        }
    return items


def kappa(pairs: list[tuple[int, int]], weights: str | None) -> float:
    n = len(pairs)
    observed = defaultdict(int)
    row, col = defaultdict(int), defaultdict(int)
    for a, b in pairs:
        observed[(a, b)] += 1
        row[a] += 1
        col[b] += 1
    span = LEVELS[-1] - LEVELS[0]

    def weight(i: int, j: int) -> float:
        if weights is None:
            return 0.0 if i == j else 1.0
        distance = abs(i - j) / span
        return distance if weights == "linear" else distance * distance

    disagreement = sum(weight(i, j) * observed[(i, j)] / n for i in LEVELS for j in LEVELS)
    chance = sum(weight(i, j) * row[i] * col[j] / (n * n) for i in LEVELS for j in LEVELS)
    return 1.0 - disagreement / chance if chance else float("nan")


def alpha_ordinal(pairs: list[tuple[int, int]]) -> float:
    """Krippendorff's alpha with the ordinal metric, two judges, no missing values."""
    coincidence = defaultdict(float)
    for a, b in pairs:
        coincidence[(a, b)] += 1
        coincidence[(b, a)] += 1
    totals = {c: sum(coincidence[(c, k)] for k in LEVELS) for c in LEVELS}
    n = sum(totals.values())

    def delta2(c: int, k: int) -> float:
        if c == k:
            return 0.0
        low, high = min(c, k), max(c, k)
        between = sum(totals[g] for g in LEVELS if low <= g <= high)
        return (between - (totals[low] + totals[high]) / 2.0) ** 2

    observed = sum(coincidence[(c, k)] * delta2(c, k) for c in LEVELS for k in LEVELS) / n
    expected = sum(totals[c] * totals[k] * delta2(c, k) for c in LEVELS for k in LEVELS) / (n * (n - 1))
    return 1.0 - observed / expected if expected else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in range(start, end + 1):
            result[order[index]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return result


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def spearman(pairs: list[tuple[int, int]]) -> float:
    return pearson(ranks([a for a, _ in pairs]), ranks([b for _, b in pairs]))


def describe(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    return {
        "items": n,
        "exact_agreement": sum(1 for a, b in pairs if a == b) / n,
        "within_one": sum(1 for a, b in pairs if abs(a - b) <= 1) / n,
        "cohen_kappa": kappa(pairs, None),
        "kappa_linear": kappa(pairs, "linear"),
        "kappa_quadratic": kappa(pairs, "quadratic"),
        "krippendorff_alpha_ordinal": alpha_ordinal(pairs),
        "spearman_rho": spearman(pairs),
        "mean_this_judge": sum(a for a, _ in pairs) / n,
        "mean_gpt_6_1_terra": sum(b for _, b in pairs) / n,
        "this_judge_higher": sum(1 for a, b in pairs if a > b),
        "this_judge_lower": sum(1 for a, b in pairs if a < b),
    }


def interval(by_task: dict[str, list[tuple[int, int]]], statistic, resamples: int = 2000, seed: int = 17) -> list[float]:
    rng = random.Random(seed)
    tasks = sorted(by_task)
    values = []
    for _ in range(resamples):
        sample = [pair for task in rng.choices(tasks, k=len(tasks)) for pair in by_task[task]]
        value = statistic(sample)
        if value == value:  # skip NaN from a degenerate resample
            values.append(value)
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evals", type=Path, default=HERE / "results" / "claude-opus-5")
    parser.add_argument("--reference", type=Path, default=mt.REPO / "results" / "final")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    mine, reference = load_items(args.evals.resolve()), load_items(args.reference.resolve())
    if set(mine) != set(reference):
        sys.exit("the two judges did not score the same items")
    if any((mine[k]["score"] == 0) != (reference[k]["score"] == 0) for k in mine):
        sys.exit("the two runs disagree on which items the grounding rule scores 0")
    judged = [k for k in mine if mine[k]["score"] != 0]

    def pairs_where(test) -> list[tuple[int, int]]:
        return [(mine[k]["score"], reference[k]["score"]) for k in judged if test(mine[k])]

    def by_task_where(test) -> dict[str, list[tuple[int, int]]]:
        grouped = defaultdict(list)
        for k in judged:
            if test(mine[k]):
                grouped[mine[k]["task"]].append((mine[k]["score"], reference[k]["score"]))
        return grouped

    report = {
        "items_scored": len(mine),
        "excluded_zero_by_grounding_rule": len(mine) - len(judged),
        "overall": describe(pairs_where(lambda item: True)),
        "by_component": {},
        "by_pattern": {},
    }
    everything = by_task_where(lambda item: True)
    report["overall"]["kappa_quadratic_ci95"] = interval(everything, lambda sample: kappa(sample, "quadratic"))
    report["overall"]["krippendorff_alpha_ordinal_ci95"] = interval(everything, alpha_ordinal)
    report["overall"]["tasks"] = len(everything)
    for component in COMPONENTS.values():
        report["by_component"][component] = describe(pairs_where(lambda item: item["component"] == component))
        report["by_component"][component]["kappa_quadratic_ci95"] = interval(
            by_task_where(lambda item: item["component"] == component), lambda sample: kappa(sample, "quadratic"))
    for pattern in mt.PATTERNS:
        report["by_pattern"][pattern] = describe(pairs_where(lambda item: item["pattern"] == pattern))
    confusion = [[sum(1 for a, b in pairs_where(lambda item: True) if a == i and b == j) for j in LEVELS] for i in LEVELS]
    report["confusion_rows_this_judge_columns_gpt"] = confusion

    table_a, _ = mt.collect(args.evals.resolve())
    table_b, _ = mt.collect(args.reference.resolve())
    cells = [(d, b, p) for d, _, _ in mt.MODELS for b in mt.METRICS for p in mt.PATTERNS]
    xs = [float(table_a[d][(b, p)]) for d, b, p in cells]
    ys = [float(table_b[d][(b, p)]) for d, b, p in cells]
    report["table_cells"] = {"cells": len(cells), "pearson_r": pearson(xs, ys), "spearman_rho": pearson(ranks(xs), ranks(ys)),
                             "mean_absolute_difference": sum(abs(x - y) for x, y in zip(xs, ys)) / len(cells)}

    def line(name: str, d: dict) -> str:
        ci = d.get("kappa_quadratic_ci95")
        return (f"{name:8s} n={d['items']:3d}  exact {d['exact_agreement']:.0%}  within-1 {d['within_one']:.0%}  "
                f"kappa {d['cohen_kappa']:.2f}  linear {d['kappa_linear']:.2f}  quadratic {d['kappa_quadratic']:.2f}"
                + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                + f"  alpha {d['krippendorff_alpha_ordinal']:.2f}  rho {d['spearman_rho']:.2f}  "
                f"means {d['mean_this_judge']:.2f} vs {d['mean_gpt_6_1_terra']:.2f}")

    print(f"items {report['items_scored']}, excluded as grounding zeros {report['excluded_zero_by_grounding_rule']}, "
          f"tasks {report['overall']['tasks']}")
    print(line("overall", report["overall"]))
    print(f"         alpha 95% interval {report['overall']['krippendorff_alpha_ordinal_ci95']}")
    for name, d in report["by_component"].items():
        print(line(name, d))
    for name, d in report["by_pattern"].items():
        print(line(name, d))
    print("confusion (rows: this judge 1..5, columns: GPT-6.1 Terra 1..5):")
    for level, row in zip(LEVELS, confusion):
        print(f"   {level}: " + " ".join(f"{count:4d}" for count in row))
    print(f"table cells: {report['table_cells']}")
    if args.write:
        target = args.evals.resolve() / "agreement_with_gpt-6.1-terra.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
