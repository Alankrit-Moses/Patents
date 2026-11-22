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
from .task_inputs import materialize_generator_inputs


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
    scale = os.getenv("EVALUATION_SCALE", "5")
    if scale not in {"3", "5"}:
        raise ValueError("EVALUATION_SCALE must be 3 or 5")
    if scale == "5":
        prompt += ("\n\nUse the following finer scale instead of the 1–3 labels above: "
                   "5 = fully correct and clear; 4 = substantially correct with a minor omission or ambiguity; "
                   "3 = partially correct with a material omission or uncertainty; "
                   "2 = weakly related with major omissions or mismatch; 1 = incorrect or unrelated. "
                   "Apply these levels to the same task-specific criteria above, including acceptance of qualitative prose.")
    judged = _judge_call(
        client,
        prompt
        + '\n\nReturn only a JSON object with exactly one key named "score", whose value '
        f"is the integer from 1 to {scale} that you selected. Do not return a rationale.",
    )
    score = judged.get("score")
    if type(score) is not int or not 1 <= score <= int(scale):
        raise ValueError(f"Judge must return an integer score from 1 to {scale}")
    return score


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
                f"""Evaluate whether this source passage instantiates the mathematical pattern M.

Mathematical pattern M:
{m_text}

Candidate passage:
{recovered}

Score 3 if the passage clearly supports the essential conditions of M; 2 if it supports some essential conditions but another is missing or uncertain; 1 if it does not meaningfully instantiate M. Accept clear qualitative prose as evidence: exact numbers are not required. Do not infer a widening absolute gap merely from different growth rates when the passage does not establish it. Judge the passage on its meaning and the supplied M.""",
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
    tab_path = resolve_safe_input(config.project_root, task["e_tab_path"], {".csv"})
    report_path = resolve_safe_input(config.project_root, task["report_path"], {".txt"})
    csv_text = read_text(tab_path)
    report = read_text(report_path)
    recovered = _recover_or_snap(predicted, report)
    if recovered is None:
        return {"E_text_alignment_score": 0}
    score = _alignment_score(
        client,
        f"""Evaluate whether the source passage describes the same specific instance represented by the table.

E.tab:
{csv_text}

Candidate passage:
{recovered}

Score 3 if it clearly describes the same instance without material contradiction; 2 if it plausibly describes the same instance but a distinguishing feature is missing or uncertain; 1 if it describes another instance or is unrelated. Identify the instance using the relevant entities, metric, period and trend as needed. A clear qualitative description can receive full credit; it need not repeat every number or use the table's wording.""",
    )
    return {"E_text_alignment_score": score}


def _moderate_m_score(client: OpenAICompatibleClient, examples: str, generated: str) -> int:
    prompt = f"""Evaluate the generated pattern definition using the examples.

Look for a meaningful structural relationship shared by the examples. The definition may use words, mathematical notation, or both. Different valid generalizations are acceptable.

Focus on what the definition gets right. Do not lower the score solely because it omits incidental details, uses different terminology, or could be more general. However, consider whether its essential conditions are supported across the examples and whether it makes a material claim that conflicts with them.

Examples:
{examples}

Generated M:
{generated}

Score:
5 = clearly identifies a meaningful pattern supported across the examples;
4 = identifies a largely supported shared pattern, with minor ambiguity or an unnecessary restriction;
3 = identifies a relevant shared relationship, but has a material omission, overly restrictive condition, or mismatch with some examples;
2 = has a limited meaningful connection to the examples;
1 = does not identify a pattern related to the examples.

Return only {{"score": <integer from 1 to 5>}}."""
    score = _judge_call(client, prompt).get("score")
    if type(score) is not int or not 1 <= score <= 5:
        raise ValueError("Judge must return an integer score from 1 to 5")
    return score


def _evaluate_exp3(
    client: OpenAICompatibleClient,
    config: HarnessConfig,
    task: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    generated = result.get("parsed_output", {})
    pairs = materialize_generator_inputs(config.project_root, task)["paired_examples"]
    examples = "\n\n".join(
        f"Example {pair['example_id']}:\nE.text:\n{pair['E.text']}\nE.tab:\n{pair['E.tab']}"
        for pair in pairs
    )
    generated_m = str(generated.get("M") or "").strip()
    generated_t = str(generated.get("T") or "").strip()
    common = """Evaluate the proposed shared pattern against the examples. Allow multiple valid generalizations. Consider whether it meaningfully explains the examples and is sufficiently specific, without requiring every detail of every example. Score 3 for a clear, meaningful shared pattern supported by the examples; 2 for a plausible pattern with a material omission, ambiguity or mismatch; 1 for an incorrect or non-meaningful shared pattern."""
    if not generated_m:
        m_score = 0
    elif os.getenv("Q3_M_RUBRIC", "moderate") == "moderate" and os.getenv("EVALUATION_SCALE", "5") == "5":
        m_score = _moderate_m_score(client, examples, generated_m)
    elif os.getenv("Q3_M_RUBRIC", "moderate") in {"moderate", "original"}:
        m_score = _alignment_score(client, f"""{common}
For M, also require an operational mathematical rule; a single example or table alone is not a general rule.

Examples:
{examples}

Generated M:
{generated_m}""")
    else:
        raise ValueError("Q3_M_RUBRIC must be moderate or original")
    t_score = _alignment_score(client, f"""{common}
For T, assess the clarity of the natural-language description.

Examples:
{examples}

Generated T:
{generated_t}""") if generated_t else 0
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
        "evaluation_rubric": f"example_alignment_0_{os.getenv('EVALUATION_SCALE', '5')}_" + ("moderate_m_v2" if os.getenv("EVALUATION_SCALE", "5") == "5" and os.getenv("Q3_M_RUBRIC", "moderate") == "moderate" else "v1"),
        "judge_model": config.judge.model,
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
