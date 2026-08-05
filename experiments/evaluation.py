"""Ground outputs in source reports and evaluate them with the study judge."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig
from .io_utils import append_jsonl, extract_json, load_jsonl, read_text, resolve_safe_input
from .reports import recover_span_canonical, snap_to_source
from .task_construction import extract_saved_excerpt


def _recover_or_snap(excerpt: str, report: str) -> str | None:
    """Canonical verbatim recovery; if that fails and GROUNDING_SNAP_THRESHOLD is
    set, fuzzy-snap the excerpt to the real report passage it matches."""
    recovered = recover_span_canonical(excerpt, report)
    if recovered is not None:
        return recovered
    threshold = os.getenv("GROUNDING_SNAP_THRESHOLD")
    if threshold:
        return snap_to_source(excerpt, report, float(threshold))
    return None


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
        recovered = _recover_or_snap(excerpt, report)
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
    recovered = _recover_or_snap(predicted, report)
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


def _judge_one(
    result: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    client: OpenAICompatibleClient,
    config: HarnessConfig,
) -> dict[str, Any]:
    base = {
        "task_id": result.get("task_id", result.get("example_id")),
        "experiment": result.get("experiment"),
        "setup": result.get("setup"),
        "pattern_id": result.get("pattern_id"),
        "metrics": {},
        "errors": [],
    }
    # Preserve current sampling metadata and historical released fields so records can be
    # grouped without rewriting their original metadata.
    for name in (
        "run_type",
        "sampling_run_id",
        "sampling_prompt_version",
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
    return base


def evaluate_results(
    results_path: Path,
    tasks_path: Path,
    output_path: Path,
    config: HarnessConfig,
    concurrency: int = 8,
) -> list[dict[str, Any]]:
    tasks = {task["task_id"]: task for task in load_jsonl(tasks_path)}
    client = OpenAICompatibleClient(config.judge)
    results = list(load_jsonl(results_path))

    # Each record makes several independent judge API calls; the judge client is
    # stateless per call, so records can be scored concurrently. Only the JSONL
    # append needs a lock. Output order is irrelevant to summarize (it groups by
    # experiment/setup), so records are written as they complete.
    if concurrency <= 1:
        evaluations = [_judge_one(result, tasks, client, config) for result in results]
        for base in evaluations:
            append_jsonl(output_path, base)
        return evaluations

    evaluations: list[dict[str, Any]] = []
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_judge_one, result, tasks, client, config) for result in results]
        for future in as_completed(futures):
            base = future.result()
            with write_lock:
                append_jsonl(output_path, base)
                evaluations.append(base)
    return evaluations
