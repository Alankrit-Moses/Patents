from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig
from .io_utils import append_jsonl, extract_json, load_jsonl, normalized_text, read_text, resolve_safe_input
from .manifest import extract_saved_excerpt, extract_saved_excerpts
from .reports import recover_exact_span


def _token_f1(left: str, right: str) -> float:
    left_tokens = re.findall(r"\w+", left.casefold())
    right_tokens = re.findall(r"\w+", right.casefold())
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts: dict[str, int] = defaultdict(int)
    right_counts: dict[str, int] = defaultdict(int)
    for token in left_tokens:
        left_counts[token] += 1
    for token in right_tokens:
        right_counts[token] += 1
    overlap = sum(min(count, right_counts[token]) for token, count in left_counts.items())
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _judge_call(client: OpenAICompatibleClient, user: str) -> dict[str, Any]:
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You are an evaluation judge. You are blind to the generation setup. Apply the rubric strictly and return JSON only.",
            },
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if not isinstance(parsed, dict):
        raise ValueError("Judge did not return a JSON object")
    parsed["judge_raw_output"] = response.text
    parsed["judge_usage"] = response.usage
    return parsed


def _find_task(tasks: dict[str, dict[str, Any]], result: dict[str, Any]) -> dict[str, Any]:
    task_id = result.get("task_id") or result.get("example_id")
    if task_id not in tasks:
        raise KeyError(f"No task record for {task_id}")
    return tasks[task_id]


def _excerpts(parsed_output: dict[str, Any]) -> list[str]:
    if "examples" in parsed_output and isinstance(parsed_output["examples"], list):
        return [
            str(item.get("excerpt", ""))
            for item in parsed_output["examples"]
            if isinstance(item, dict)
        ]
    if "excerpt" in parsed_output:
        return [str(parsed_output.get("excerpt", ""))]
    return []


def _evaluate_exp1(
    client: OpenAICompatibleClient,
    config: HarnessConfig,
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    report = read_text(resolve_safe_input(config.project_root, task["report_path"], {".txt"}))
    m_text = read_text(resolve_safe_input(config.project_root, task["m_path"], {".txt"}))
    excerpts = _excerpts(result.get("parsed_output", {}))
    exact_flags = [recover_exact_span(excerpt, report) is not None for excerpt in excerpts]
    judged = _judge_call(
        client,
        f"""Mathematical pattern definition:
{m_text}

Candidate exact excerpts:
{json.dumps(excerpts, ensure_ascii=False, indent=2)}

Score each candidate from 1 to 5 for satisfying the definition. A 5 is a clear, specific instance; a 4 is valid with minor ambiguity; 1-3 are invalid or weak. Do not assess recall because gold is non-exhaustive.

Return {{"candidate_scores":[{{"score":1,"rationale":""}}],"mean_judge_score":0.0}}.""",
    )
    scores = [
        max(1, min(5, int(item.get("score", 1))))
        for item in judged.get("candidate_scores", [])
        if isinstance(item, dict)
    ]
    return {
        "mean_judge_score": mean(scores) if scores else 0.0,
        "valid_at_3": sum(score >= 4 for score in scores[:3]) / 3,
        "strong_at_3": sum(score == 5 for score in scores[:3]) / 3,
        "exact_quote_compliance": sum(exact_flags[:3]) / 3,
        "count_compliance": int(len(excerpts) == 3),
        "candidate_scores": scores,
        "judge_details": judged,
    }


def _evaluate_exp2(
    client: OpenAICompatibleClient,
    config: HarnessConfig,
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    predicted = (_excerpts(result.get("parsed_output", {})) or [""])[0]
    gold_path = resolve_safe_input(config.project_root, task["gold_e_text_path"], {".txt"})
    tab_path = resolve_safe_input(config.project_root, task["e_tab_path"], {".csv"})
    report_path = resolve_safe_input(config.project_root, task["report_path"], {".txt"})
    gold_references = extract_saved_excerpts(gold_path)
    csv_text = read_text(tab_path)
    report = read_text(report_path)
    judged = _judge_call(
        client,
        f"""Provided compact E.tab:
{csv_text}

Saved reference E.text span(s):
{json.dumps(gold_references, ensure_ascii=False, indent=2)}

Candidate E.text:
{predicted}

Judge whether the candidate describes the same table evidence. Return integer 0 or 1 for entity_correctness, metric_correctness, interval_correctness, and pattern_behavior_correctness, plus alignment_score from 1 to 5 and a short rationale.

Return {{"entity_correctness":0,"metric_correctness":0,"interval_correctness":0,"pattern_behavior_correctness":0,"alignment_score":1,"rationale":""}}.""",
    )
    return {
        "sequence_overlap": max(
            (
                SequenceMatcher(
                    None, normalized_text(predicted), normalized_text(reference)
                ).ratio()
                for reference in gold_references
            ),
            default=0.0,
        ),
        "token_f1": max(
            (_token_f1(predicted, reference) for reference in gold_references),
            default=0.0,
        ),
        "exact_quote_compliance": int(recover_exact_span(predicted, report) is not None),
        "entity_correctness": int(judged.get("entity_correctness", 0)),
        "metric_correctness": int(judged.get("metric_correctness", 0)),
        "interval_correctness": int(judged.get("interval_correctness", 0)),
        "pattern_behavior_correctness": int(
            judged.get("pattern_behavior_correctness", 0)
        ),
        "judge_alignment_score": max(
            1, min(5, int(judged.get("alignment_score", 1)))
        ),
        "judge_details": judged,
    }


def _load_pairs(config: HarnessConfig, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for example in examples:
        text_path = resolve_safe_input(config.project_root, example["e_text_path"], {".txt"})
        tab_path = resolve_safe_input(config.project_root, example["e_tab_path"], {".csv"})
        pairs.append(
            {
                "example_id": example["example_id"],
                "E.text": extract_saved_excerpt(text_path),
                "E.tab": read_text(tab_path),
            }
        )
    return pairs


def _ready_negatives(pattern_id: str) -> list[dict[str, Any]]:
    path = Path(__file__).parent / "hard_negatives" / f"{pattern_id}.jsonl"
    return [record for record in load_jsonl(path) if record.get("status") == "ready"]


def _evaluate_exp3(
    client: OpenAICompatibleClient,
    config: HarnessConfig,
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    generated = result.get("parsed_output", {})
    canonical_m = read_text(resolve_safe_input(config.project_root, task["m_path"], {".txt"}))
    canonical_t = read_text(resolve_safe_input(config.project_root, task["t_path"], {".txt"}))
    heldout = _load_pairs(config, task.get("heldout_examples", []))
    negatives = _ready_negatives(task["pattern_id"])
    judged = _judge_call(
        client,
        f"""Canonical pattern card (reference, not a required wording template):
M:
{canonical_m}

T:
{canonical_t}

Generated pattern card:
{json.dumps(generated, ensure_ascii=False, indent=2)}

Held-out positive pairs:
{json.dumps(heldout, ensure_ascii=False, indent=2)}

Hard negatives (may be empty):
{json.dumps(negatives, ensure_ascii=False, indent=2)}

Evaluate semantic fidelity rather than wording similarity. Return 1-5 scores for M, T, and operationalizability; held_out_positive_acceptance as a 0-1 rate; hard_negative_rejection as a 0-1 rate or null when there are no ready negatives; and a rationale.

Return {{"M_score":1,"T_score":1,"operationalizability":1,"held_out_positive_acceptance":0.0,"hard_negative_rejection":null,"rationale":""}}.""",
    )
    return {
        "M_judge_score": max(1, min(5, int(judged.get("M_score", 1)))),
        "T_judge_score": max(1, min(5, int(judged.get("T_score", 1)))),
        "operationalizability": max(
            1, min(5, int(judged.get("operationalizability", 1)))
        ),
        "held_out_positive_acceptance": float(
            judged.get("held_out_positive_acceptance", 0.0)
        ),
        "hard_negative_rejection": judged.get("hard_negative_rejection"),
        "judge_details": judged,
    }


def evaluate_results(
    results_path: Path,
    tasks_path: Path,
    output_path: Path,
    config: HarnessConfig,
) -> list[dict[str, Any]]:
    tasks = {task["task_id"]: task for task in load_jsonl(tasks_path)}
    client = OpenAICompatibleClient(config.judge)
    evaluations: list[dict[str, Any]] = []
    for result in load_jsonl(results_path):
        base = {
            "task_id": result.get("task_id", result.get("example_id")),
            "experiment": result.get("experiment"),
            "setup": result.get("setup"),
            "pattern_id": result.get("pattern_id"),
            "metrics": {},
            "errors": [],
        }
        try:
            task = _find_task(tasks, result)
            experiment = str(result["experiment"])
            if result.get("errors"):
                raise ValueError("Generator run failed: " + "; ".join(result["errors"]))
            if experiment == "1":
                base["metrics"] = _evaluate_exp1(client, config, task, result)
            elif experiment == "2":
                base["metrics"] = _evaluate_exp2(client, config, task, result)
            elif experiment == "3":
                base["metrics"] = _evaluate_exp3(client, config, task, result)
            else:
                raise ValueError(f"Unknown experiment: {experiment}")
        except Exception as error:
            base["errors"].append(str(error))
        append_jsonl(output_path, base)
        evaluations.append(base)
    return evaluations


def summarize_evaluations(path: Path) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(path):
        if not record.get("errors"):
            groups[(str(record["experiment"]), record["setup"])].append(record["metrics"])
    summary: dict[str, Any] = {}
    for (experiment, setup), metrics_list in sorted(groups.items()):
        numeric: dict[str, list[float]] = defaultdict(list)
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric[key].append(float(value))
        summary[f"experiment_{experiment}/setup_{setup}"] = {
            "runs": len(metrics_list),
            "means": {key: mean(values) for key, values in sorted(numeric.items()) if values},
        }
    return summary
