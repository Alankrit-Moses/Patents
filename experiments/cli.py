from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent import setup_c_plan_preview
from .config import HarnessConfig, load_config
from .io_utils import stable_hash, write_jsonl
from .judge import evaluate_results, summarize_evaluations
from .manifest import build_manifest, build_tasks, manifest_counts, save_manifest_and_tasks
from .prompts import build_prompt, generator_input_refs, materialize_generator_inputs
from .runner import run_tasks


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_corpus(config: HarnessConfig):
    manifest = build_manifest(config)
    tasks = build_tasks(manifest, config)
    return manifest, tasks


def _filter_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = tasks
    if getattr(args, "experiment", "all") != "all":
        selected = [task for task in selected if task["experiment"] == args.experiment]
    if getattr(args, "pattern", None):
        selected = [task for task in selected if task["pattern_id"] == args.pattern]
    if getattr(args, "task_id", None):
        selected = [task for task in selected if task["task_id"] == args.task_id]
    if getattr(args, "limit", None) is not None:
        selected = selected[: args.limit]
    return selected


def command_manifest(args: argparse.Namespace, config: HarnessConfig) -> int:
    manifest, tasks = _load_corpus(config)
    output_dir = Path(args.output).resolve() if args.output else config.output_dir
    manifest_path, tasks_path = save_manifest_and_tasks(manifest, tasks, output_dir)
    print(json.dumps(manifest_counts(manifest, tasks), indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Tasks: {tasks_path}")
    return 0


def command_dry_run(args: argparse.Namespace, config: HarnessConfig) -> int:
    manifest, tasks = _load_corpus(config)
    output_dir = Path(args.output).resolve() if args.output else config.output_dir
    manifest_path, tasks_path = save_manifest_and_tasks(manifest, tasks, output_dir)
    prompt_index: list[dict[str, Any]] = []
    errors: list[str] = []
    for task in tasks:
        try:
            inputs = materialize_generator_inputs(config.project_root, task)
            for setup in ("A", "B"):
                messages = build_prompt(setup, task, inputs)
                serialized = json.dumps(messages, ensure_ascii=False)
                prompt_index.append(
                    {
                        "task_id": task["task_id"],
                        "experiment": task["experiment"],
                        "setup": setup,
                        "inputs_used": generator_input_refs(task),
                        "message_count": len(messages),
                        "prompt_chars": sum(len(message["content"]) for message in messages),
                        "prompt_sha256": stable_hash(serialized),
                        "user_prompt_preview": messages[-1]["content"][:500],
                    }
                )
            prompt_index.append(
                {
                    "task_id": task["task_id"],
                    "experiment": task["experiment"],
                    "setup": "C",
                    "inputs_used": generator_input_refs(task),
                    "planned_actions": setup_c_plan_preview(task),
                }
            )
        except Exception as error:
            errors.append(f"{task['task_id']}: {error}")
    prompt_path = output_dir / "dry_run_prompt_index.jsonl"
    write_jsonl(prompt_path, prompt_index)
    counts = manifest_counts(manifest, tasks)
    counts["dry_run_prompt_records"] = len(prompt_index)
    counts["dry_run_errors"] = errors
    print(json.dumps(counts, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Tasks: {tasks_path}")
    print(f"Prompt index: {prompt_path}")
    print("LLM calls: 0")
    return 1 if errors else 0


def command_run(args: argparse.Namespace, config: HarnessConfig) -> int:
    manifest, tasks = _load_corpus(config)
    save_manifest_and_tasks(manifest, tasks, config.output_dir)
    tasks = _filter_tasks(tasks, args)
    if not tasks:
        print("No matching tasks.")
        return 1
    setups = ["A", "B", "C"] if args.setup == "all" else [args.setup.upper()]
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.output_dir / "runs" / f"results-{_timestamp()}.jsonl"
    )
    run_tasks(tasks, setups, config, output_path)
    print(f"Results: {output_path}")
    return 0


def command_judge(args: argparse.Namespace, config: HarnessConfig) -> int:
    results_path = Path(args.results).resolve()
    tasks_path = (
        Path(args.tasks).resolve() if args.tasks else config.output_dir / "tasks.jsonl"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.output_dir / "evaluations" / f"evaluations-{_timestamp()}.jsonl"
    )
    evaluate_results(results_path, tasks_path, output_path, config)
    print(f"Evaluations: {output_path}")
    return 0


def command_summarize(args: argparse.Namespace, _config: HarnessConfig) -> int:
    summary = summarize_evaluations(Path(args.evaluations).resolve())
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WIPO modular-framework experiment harness")
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Build corpus manifest and tasks")
    manifest_parser.add_argument("--output", help="Output directory")

    dry_parser = subparsers.add_parser("dry-run", help="Scan data and build prompts without LLM calls")
    dry_parser.add_argument("--output", help="Output directory")

    run_parser = subparsers.add_parser("run", help="Run generator experiments")
    run_parser.add_argument("--setup", choices=["A", "B", "C", "all"], default="all")
    run_parser.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    run_parser.add_argument("--pattern", help="Filter by pattern ID, e.g. P1")
    run_parser.add_argument("--task-id", help="Run one task ID")
    run_parser.add_argument("--limit", type=int, help="Limit selected task count")
    run_parser.add_argument("--output", help="Result JSONL path")

    judge_parser = subparsers.add_parser("judge", help="Blind-judge result JSONL")
    judge_parser.add_argument("--results", required=True, help="Generator result JSONL")
    judge_parser.add_argument("--tasks", help="Task JSONL; defaults to artifacts/tasks.jsonl")
    judge_parser.add_argument("--output", help="Evaluation JSONL path")

    summary_parser = subparsers.add_parser("summarize", help="Aggregate evaluation metrics")
    summary_parser.add_argument("--evaluations", required=True, help="Evaluation JSONL")
    summary_parser.add_argument("--output", help="Optional summary JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    commands = {
        "manifest": command_manifest,
        "dry-run": command_dry_run,
        "run": command_run,
        "judge": command_judge,
        "summarize": command_summarize,
    }
    return commands[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
