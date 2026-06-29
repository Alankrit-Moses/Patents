from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .agent import setup_c_plan_preview
from .config import load_config
from .io_utils import read_text, resolve_safe_input
from .manifest import extract_saved_excerpt
from .prompts import build_prompt, generator_input_refs, materialize_generator_inputs
from .reports import recover_exact_span


JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge blind to the generation setup. Reason internally, "
    "apply the alignment rubric strictly, and output only the requested JSON score."
)

JUDGE_JSON_SUFFIX = (
    '\n\nReturn only a JSON object with exactly one key named "score", whose value '
    "is the integer from 1 to 5 that you selected. Do not return a rationale."
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _latest_jsonl(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    return candidates[-1]


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("._")
    return value or "item"


def _format_messages(messages: list[dict[str, str]]) -> str:
    blocks = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        blocks.append(f"===== MESSAGE {index}: {role.upper()} =====\n{content}")
    return "\n\n".join(blocks) + "\n"


def _setup_c_input_text(task: dict[str, Any], inputs: dict[str, Any]) -> str:
    packet = {
        "note": (
            "Setup C is not a single-shot prompt. This file records the typed "
            "input packet given to the framework-compiled orchestrator: the "
            "framework query, task-visible evidence, and planned reusable actions."
        ),
        "task_id": task["task_id"],
        "experiment": task["experiment"],
        "pattern_id": task["pattern_id"],
        "framework_query": task["framework_query"],
        "inputs_used": generator_input_refs(task),
        "planned_actions": setup_c_plan_preview(task),
        "materialized_inputs": inputs,
    }
    return json.dumps(packet, ensure_ascii=False, indent=2) + "\n"


def _output_text(record: dict[str, Any]) -> str:
    payload = {
        "task_id": record.get("task_id"),
        "experiment": record.get("experiment"),
        "setup": record.get("setup"),
        "pattern_id": record.get("pattern_id"),
        "example_id": record.get("example_id"),
        "errors": record.get("errors", []),
        "raw_output": record.get("raw_output", ""),
        "parsed_output": record.get("parsed_output", {}),
    }
    if record.get("setup") == "C":
        payload["agent_trace"] = record.get("agent_trace", [])
        payload["selected_chunks"] = record.get("selected_chunks", [])
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


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


def _format_judge_messages(user_prompt: str) -> str:
    return _format_messages(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + JUDGE_JSON_SUFFIX},
        ]
    )


def _evaluator_prompt_text(
    root: Path,
    task: dict[str, Any] | None,
    result: dict[str, Any],
) -> str:
    if task is None:
        return "Could not reconstruct evaluator prompt: no matching task record was found.\n"
    if result.get("errors"):
        return (
            "No judge prompt was issued for semantic alignment because the generator "
            "record contains errors:\n"
            + json.dumps(result.get("errors", []), ensure_ascii=False, indent=2)
            + "\n"
        )

    experiment = str(result.get("experiment"))
    parsed_output = result.get("parsed_output", {})

    if experiment == "1":
        report = read_text(resolve_safe_input(root, task["report_path"], {".txt"}))
        m_text = read_text(resolve_safe_input(root, task["m_path"], {".txt"}))
        excerpts = _excerpts(parsed_output)
        target_count = int(task.get("target_count", 3))
        blocks = []
        for index in range(max(target_count, len(excerpts))):
            excerpt = excerpts[index] if index < len(excerpts) else ""
            recovered = recover_exact_span(excerpt, report)
            header = f"===== EVALUATOR CALL {index + 1} / EXPERIMENT 1 ====="
            if recovered is None:
                blocks.append(
                    header
                    + "\nNo LLM judge call was issued. The candidate failed exact "
                    "membership recovery against the report, so the assigned score is 0.\n\n"
                    "Candidate text:\n"
                    + excerpt
                )
                continue
            user_prompt = f"""Score how well this exact report excerpt instantiates the supplied mathematical pattern definition.

Mathematical pattern definition:
{m_text}

Candidate report excerpt:
{recovered}

Use this alignment scale: 5 = clear and complete instantiation; 4 = valid instantiation with minor ambiguity or omitted context; 3 = partial or uncertain instantiation; 2 = weakly related but missing central conditions; 1 = not an instantiation. Judge semantic pattern alignment, not wording similarity. The saved example collection is non-exhaustive and must not be considered."""
            blocks.append(header + "\n" + _format_judge_messages(user_prompt))
        return "\n\n".join(blocks) + "\n"

    if experiment == "2":
        predicted = (_excerpts(parsed_output) or [""])[0]
        gold_path = resolve_safe_input(root, task["gold_e_text_path"], {".txt"})
        tab_path = resolve_safe_input(root, task["e_tab_path"], {".csv"})
        report_path = resolve_safe_input(root, task["report_path"], {".txt"})
        gold_text = extract_saved_excerpt(gold_path)
        csv_text = read_text(tab_path)
        report = read_text(report_path)
        recovered = recover_exact_span(predicted, report)
        if recovered is None:
            return (
                "No LLM judge call was issued. The candidate failed exact membership "
                "recovery against the report, so the assigned score is 0.\n\n"
                "Candidate text:\n"
                + predicted
                + "\n"
            )
        user_prompt = f"""Score how well the candidate exact report excerpt describes the same evidence as the provided compact table. The paired reference excerpt is diagnostic evidence showing the intended table-text correspondence, not a wording template. A different exact report passage can receive full credit if it expresses the same entities, metric, interval, values or trend, and pattern behavior.

Provided compact E.tab:
{csv_text}

Paired reference E.text:
{gold_text}

Candidate exact report excerpt:
{recovered}

Use this alignment scale: 5 = describes the same evidence clearly and completely; 4 = same evidence with a minor omission or ambiguity; 3 = partially aligned evidence; 2 = only weakly related evidence; 1 = unrelated to the table evidence. Judge semantic evidence alignment, not lexical overlap with the reference excerpt."""
        return "===== EVALUATOR CALL 1 / EXPERIMENT 2 =====\n" + _format_judge_messages(
            user_prompt
        )

    if experiment == "3":
        generated = parsed_output
        canonical_m = read_text(resolve_safe_input(root, task["m_path"], {".txt"}))
        canonical_t = read_text(resolve_safe_input(root, task["t_path"], {".txt"}))
        generated_m = str(generated.get("M", ""))
        generated_t = str(generated.get("T", ""))
        m_prompt = f"""Score the semantic and operational alignment of a generated mathematical pattern definition with the canonical mathematical definition. Equivalent notation and wording are acceptable; a specific example, CSV/table, or non-general rule is not a valid replacement.

Canonical M:
{canonical_m}

Generated M:
{generated_m}

Use this alignment scale: 5 = equivalent operational rule; 4 = substantially aligned with a minor omission or ambiguity; 3 = captures part of the rule; 2 = weakly related; 1 = not aligned."""
        t_prompt = f"""Score the semantic alignment of a generated natural-language pattern description with the canonical description. Equivalent wording is acceptable; a specific example or vague topical description is not a valid replacement.

Canonical T:
{canonical_t}

Generated T:
{generated_t}

Use this alignment scale: 5 = equivalent clear description; 4 = substantially aligned with a minor omission or ambiguity; 3 = captures part of the pattern; 2 = weakly related; 1 = not aligned."""
        return (
            "===== EVALUATOR CALL 1 / EXPERIMENT 3: M =====\n"
            + _format_judge_messages(m_prompt)
            + "\n\n===== EVALUATOR CALL 2 / EXPERIMENT 3: T =====\n"
            + _format_judge_messages(t_prompt)
        )

    return f"Could not reconstruct evaluator prompt: unknown experiment {experiment}.\n"


def _load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    return {record["task_id"]: record for record in _read_jsonl(path)}


def _evaluation_key(record: dict[str, Any]) -> tuple[str, str, str]:
    sample_key = record.get("sample_id")
    if sample_key is None and "sample_index" in record:
        sample_key = f"sample_index={record.get('sample_index')}"
    return str(record.get("task_id")), str(record.get("setup")), str(sample_key or "")


def export_input_output(
    *,
    config_path: Path | None,
    results_path: Path | None,
    evaluations_path: Path | None,
    tasks_path: Path | None,
    output_dir: Path | None,
    condition: str,
    clean: bool,
) -> Path:
    config = load_config(str(config_path) if config_path else None)
    artifacts_dir = config.output_dir
    results_path = results_path or _latest_jsonl(artifacts_dir / "runs", "results-*.jsonl")
    evaluations_path = evaluations_path or _latest_jsonl(
        artifacts_dir / "evaluations", "evaluations-[0-9]*.jsonl"
    )
    tasks_path = tasks_path or artifacts_dir / "tasks.jsonl"
    output_dir = output_dir or artifacts_dir / "input_output"
    condition_dir_name = _safe_name(condition)

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _load_tasks(tasks_path)
    results = _read_jsonl(results_path)
    evaluations = {_evaluation_key(record): record for record in _read_jsonl(evaluations_path)}

    manifest_rows: list[dict[str, Any]] = []
    missing_tasks: list[str] = []
    missing_evaluations: list[dict[str, str]] = []

    for record in results:
        task_id = str(record.get("task_id"))
        setup = str(record.get("setup"))
        setup_dir = (
            output_dir
            / _safe_name(task_id)
            / condition_dir_name
            / f"setup_{_safe_name(setup)}"
        )
        if "sample_index" in record:
            sample_number = int(record.get("sample_index", 0)) + 1
            task_dir = setup_dir / f"sample_{sample_number}"
        else:
            task_dir = setup_dir
        task_dir.mkdir(parents=True, exist_ok=True)

        task = tasks.get(task_id)
        if task is None:
            missing_tasks.append(task_id)
            input_text = (
                "Could not reconstruct input: no matching task record was found in "
                f"{tasks_path}.\n\nResult inputs_used:\n"
                + json.dumps(record.get("inputs_used", {}), ensure_ascii=False, indent=2)
                + "\n"
            )
        else:
            inputs = materialize_generator_inputs(config.project_root, task)
            if setup in {"A", "B"}:
                input_text = _format_messages(build_prompt(setup, task, inputs))
            elif setup == "C":
                input_text = _setup_c_input_text(task, inputs)
            else:
                input_text = json.dumps(
                    {
                        "note": f"Unknown setup {setup}; writing task and materialized inputs.",
                        "task": task,
                        "inputs": inputs,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

        evaluation = evaluations.get(_evaluation_key(record))
        if evaluation is None:
            missing_evaluations.append({"task_id": task_id, "setup": setup})

        (task_dir / "input.txt").write_text(input_text, encoding="utf-8")
        (task_dir / "output.txt").write_text(_output_text(record), encoding="utf-8")
        (task_dir / "evaluator_prompt.txt").write_text(
            _evaluator_prompt_text(config.project_root, task, record), encoding="utf-8"
        )
        (task_dir / "evaluation.json").write_text(
            json.dumps(evaluation or {}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "task_id": task_id,
                "experiment": record.get("experiment"),
                "setup": setup,
                "pattern_id": record.get("pattern_id"),
                "example_id": record.get("example_id"),
                "folder": str(task_dir.relative_to(output_dir)),
                "condition": condition,
                "sample_index": record.get("sample_index"),
                "sample_id": record.get("sample_id"),
                "generator_model": record.get("generator_model"),
                "temperature": record.get("temperature"),
                "has_errors": bool(record.get("errors")),
                "has_evaluation": evaluation is not None,
            }
        )

    summary = {
        "results_path": str(results_path.resolve()),
        "evaluations_path": str(evaluations_path.resolve()),
        "tasks_path": str(tasks_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "condition": condition,
        "records_exported": len(results),
        "setups": sorted({row["setup"] for row in manifest_rows}),
        "missing_task_records": sorted(set(missing_tasks)),
        "missing_evaluations": missing_evaluations,
    }
    (output_dir / "README.md").write_text(
        "Input/output inspection export\n"
        "==============================\n\n"
        "Folder structure\n"
        "----------------\n\n"
        "This folder is organized task-first, then by run condition, then by setup:\n\n"
        "input_output/\n"
        "  <task_id>/\n"
        "    <condition>/\n"
        "      setup_A/\n"
        "        input.txt\n"
        "        output.txt\n"
        "        evaluator_prompt.txt\n"
        "        evaluation.json\n"
        "      setup_B/\n"
        "        ...\n"
        "      setup_C/\n"
        "        ...\n\n"
        "For stochastic runs with multiple samples, each setup contains sample folders:\n\n"
        "input_output/\n"
        "  <task_id>/\n"
        "    temperature_zero_point_two/\n"
        "      setup_A/\n"
        "        sample_1/\n"
        "          input.txt\n"
        "          output.txt\n"
        "          evaluator_prompt.txt\n"
        "          evaluation.json\n\n"
        "For example, the Setup A files for a deterministic temperature-zero run of task exp1-P1-genAI are stored as:\n\n"
        "input_output/exp1-P1-genAI/temperature_zero/setup_A/\n\n"
        "Files inside each setup folder\n"
        "------------------------------\n\n"
        "- input.txt: prompt sent to the setup, or Setup C's typed orchestrator input packet.\n"
        "- output.txt: full generator output that was evaluated by the judge.\n"
        "- evaluator_prompt.txt: judge prompt(s), or exact-membership gate note when no LLM judge call was made.\n"
        "- evaluation.json: raw judge evaluation record with score fields.\n\n"
        "Setups\n"
        "------\n\n"
        "- setup_A: minimal-prose single-shot baseline. It receives the task evidence in ordinary prose and returns JSON.\n"
        "- setup_B: framework-aware single-shot prompt. It receives the same task-visible evidence, but with the framework primer, typed query, tagged components, and structured output constraints.\n"
        "- setup_C: framework-compiled workflow. It receives the typed task packet, uses the common orchestrator and reusable worker actions, and returns only the requested target component(s). For Setup C, input.txt records the orchestrator input packet rather than a single LLM prompt.\n\n"
        "Experiments\n"
        "-----------\n\n"
        "- Experiment 1, M -> E.text[3]: given a mathematical pattern definition and one report, return three exact report excerpts instantiating the pattern.\n"
        "- Experiment 2, E.tab -> E.text: given one compact tabular example and the corresponding report, return the exact report excerpt describing the same evidence.\n"
        "- Experiment 3, (E.text,E.tab)^4 -> M,T: given four paired text/table examples from a pattern, induce the mathematical definition M and natural-language description T.\n\n"
        "Patterns\n"
        "--------\n\n"
        "- P1: trajectory-shift pattern, where an entity/metric changes behavior around an identifiable time or event.\n"
        "- P2: comparative-growth pattern, where comparable entities are compared by CAGR over the same interval.\n\n"
        "Conditions\n"
        "----------\n\n"
        "Condition folders separate different run settings. The deterministic run is stored under `temperature_zero`. Robustness samples at temperature 0.2 are stored under `temperature_zero_point_two`, with one `sample_<n>` folder per sampled output.\n\n"
        f"The most recently exported condition for this folder is `{condition}`.\n\n"
        "Evaluation\n"
        "----------\n\n"
        "Text-output experiments first apply an exact-membership gate: if an expected report excerpt is not recoverable verbatim from the report, the score is 0 and evaluator_prompt.txt records that no LLM judge call was made. Otherwise, the judge prompt asks for a 1-5 alignment score. Experiment 3 separately judges generated M and T against the canonical pattern definitions.\n\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps({"summary": summary, "records": manifest_rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export input/output/evaluation inspection folders")
    parser.add_argument("--config", help="Config JSON path")
    parser.add_argument("--results", help="Generator results JSONL; defaults to latest deterministic run")
    parser.add_argument("--evaluations", help="Judge evaluations JSONL; defaults to latest evaluations JSONL")
    parser.add_argument("--tasks", help="Task JSONL; defaults to artifacts/tasks.jsonl")
    parser.add_argument("--output", help="Output folder")
    parser.add_argument(
        "--condition",
        default="temperature_zero",
        help="Run-condition folder name, e.g. temperature_zero or temperature_0.2_sample_1",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not delete an existing output folder first")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = export_input_output(
        config_path=Path(args.config).resolve() if args.config else None,
        results_path=Path(args.results).resolve() if args.results else None,
        evaluations_path=Path(args.evaluations).resolve() if args.evaluations else None,
        tasks_path=Path(args.tasks).resolve() if args.tasks else None,
        output_dir=Path(args.output).resolve() if args.output else None,
        condition=args.condition,
        clean=not args.no_clean,
    )
    print(f"Input/output export: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
