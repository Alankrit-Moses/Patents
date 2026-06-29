from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig
from .io_utils import append_jsonl, extract_json, load_jsonl, read_text, resolve_safe_input
from .manifest import extract_saved_excerpt
from .reports import recover_exact_span


def _judge_call(client: OpenAICompatibleClient, user: str) -> dict[str, Any]:
    response = client.complete(
        [
            {
                "role": "system",
                "content": "You are an evaluation judge blind to the generation setup. Reason internally, apply the alignment rubric strictly, and output only the requested JSON score.",
            },
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    parsed = extract_json(response.text)
    if not isinstance(parsed, dict):
        raise ValueError("Judge did not return a JSON object")
    return parsed


def _alignment_score(client: OpenAICompatibleClient, prompt: str) -> int:
    judged = _judge_call(
        client,
        prompt
        + '\n\nReturn only a JSON object with exactly one key named "score", whose value '
        "is the integer from 1 to 5 that you selected. Do not return a rationale.",
    )
    return max(1, min(5, int(judged.get("score", 1))))


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
    target_count = int(task.get("target_count", 3))
    scores: list[int] = []
    for index in range(max(target_count, len(excerpts))):
        excerpt = excerpts[index] if index < len(excerpts) else ""
        recovered = recover_exact_span(excerpt, report)
        if recovered is None:
            scores.append(0)
            continue
        scores.append(
            _alignment_score(
                client,
                f"""Score how well this exact report excerpt instantiates the supplied mathematical pattern definition.

Mathematical pattern definition:
{m_text}

Candidate report excerpt:
{recovered}

Use this alignment scale: 5 = clear and complete instantiation; 4 = valid instantiation with minor ambiguity or omitted context; 3 = partial or uncertain instantiation; 2 = weakly related but missing central conditions; 1 = not an instantiation. Judge semantic pattern alignment, not wording similarity. The saved example collection is non-exhaustive and must not be considered.""",
            )
        )
    return {"E_text_alignment_scores": scores}


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
    gold_text = extract_saved_excerpt(gold_path)
    csv_text = read_text(tab_path)
    report = read_text(report_path)
    recovered = recover_exact_span(predicted, report)
    if recovered is None:
        return {"E_text_alignment_score": 0}
    score = _alignment_score(
        client,
        f"""Score how well the candidate exact report excerpt describes the same evidence as the provided compact table. The paired reference excerpt is diagnostic evidence showing the intended table-text correspondence, not a wording template. A different exact report passage can receive full credit if it expresses the same entities, metric, interval, values or trend, and pattern behavior.

Provided compact E.tab:
{csv_text}

Paired reference E.text:
{gold_text}

Candidate exact report excerpt:
{recovered}

Use this alignment scale: 5 = describes the same evidence clearly and completely; 4 = same evidence with a minor omission or ambiguity; 3 = partially aligned evidence; 2 = only weakly related evidence; 1 = unrelated to the table evidence. Judge semantic evidence alignment, not lexical overlap with the reference excerpt.""",
    )
    return {"E_text_alignment_score": score}


def _evaluate_exp3(
    client: OpenAICompatibleClient,
    config: HarnessConfig,
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    generated = result.get("parsed_output", {})
    canonical_m = read_text(resolve_safe_input(config.project_root, task["m_path"], {".txt"}))
    canonical_t = read_text(resolve_safe_input(config.project_root, task["t_path"], {".txt"}))
    generated_m = str(generated.get("M", ""))
    generated_t = str(generated.get("T", ""))
    m_score = _alignment_score(
        client,
        f"""Score the semantic and operational alignment of a generated mathematical pattern definition with the canonical mathematical definition. Equivalent notation and wording are acceptable; a specific example, CSV/table, or non-general rule is not a valid replacement.

Canonical M:
{canonical_m}

Generated M:
{generated_m}

Use this alignment scale: 5 = equivalent operational rule; 4 = substantially aligned with a minor omission or ambiguity; 3 = captures part of the rule; 2 = weakly related; 1 = not aligned.""",
    )
    t_score = _alignment_score(
        client,
        f"""Score the semantic alignment of a generated natural-language pattern description with the canonical description. Equivalent wording is acceptable; a specific example or vague topical description is not a valid replacement.

Canonical T:
{canonical_t}

Generated T:
{generated_t}

Use this alignment scale: 5 = equivalent clear description; 4 = substantially aligned with a minor omission or ambiguity; 3 = captures part of the pattern; 2 = weakly related; 1 = not aligned.""",
    )
    return {"M_alignment_score": m_score, "T_alignment_score": t_score}


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
        # Preserve optional sampling metadata so robustness evaluations can be
        # grouped without changing deterministic result records.
        for name in (
            "run_type",
            "robustness_run_id",
            "robustness_prompt_version",
            "b_variant",
            "context_limit_tokens",
            "generator_model",
            "temperature",
            "samples_per_condition",
            "sample_index",
            "sample_id",
            "max_attempts",
            "attempt_count",
            "failed_attempts",
        ):
            if name in result:
                base[name] = result[name]
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
                elif isinstance(value, list) and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in value
                ):
                    numeric[key].extend(float(item) for item in value)
        summary[f"experiment_{experiment}/setup_{setup}"] = {
            "runs": len(metrics_list),
            "means": {key: mean(values) for key, values in sorted(numeric.items()) if values},
        }
    return summary
