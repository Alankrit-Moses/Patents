"""Run the agentic executor once for each selected framework task."""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .executor import AgenticExecutor
from .client import OpenAICompatibleClient
from .config import HarnessConfig, load_config
from .io_utils import append_jsonl, load_jsonl
from .task_construction import build_manifest, build_tasks, save_manifest_and_tasks
from .task_inputs import generator_input_refs, materialize_generator_inputs


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def prompt_version_for() -> str:
    """Prompt tag recorded by the paper's single agentic executor."""
    return "v2-agentic"


def _base_record(task: dict[str, Any], setup: str, prompt_version: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "experiment": task["experiment"],
        "setup": setup,
        "prompt_version": prompt_version,
        "pattern_id": task["pattern_id"],
        "example_id": task.get("source_example_id", task["task_id"]),
        "framework_query": task["framework_query"],
        "inputs_used": generator_input_refs(task),
        "selected_chunks": [],
        "agent_trace": [],
        "raw_output": "",
        "parsed_output": {},
        "errors": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_output(experiment: str, parsed: dict[str, Any]) -> None:
    if experiment == "1":
        examples = parsed.get("examples")
        if not isinstance(examples, list):
            raise ValueError("Expected JSON key 'examples' with a list value")
        if len(examples) != 3:
            raise ValueError("Expected exactly 3 example objects in 'examples'")
        for index, item in enumerate(examples, start=1):
            if not isinstance(item, dict) or not isinstance(item.get("excerpt"), str):
                raise ValueError(f"Expected examples[{index}] to contain string key 'excerpt'")
    elif experiment == "2":
        if not isinstance(parsed.get("excerpt"), str):
            raise ValueError("Expected JSON key 'excerpt' with a string value")
    elif experiment == "3":
        if not isinstance(parsed.get("M"), str) or not isinstance(parsed.get("T"), str):
            raise ValueError("Expected JSON keys 'M' and 'T' with string values")
    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def run_task(
    task: dict[str, Any],
    config: HarnessConfig,
    client: OpenAICompatibleClient,
) -> dict[str, Any]:
    setup = "C"  # Historical label retained in released result records.
    record = _base_record(task, setup, prompt_version_for())
    try:
        inputs = materialize_generator_inputs(config.project_root, task)
        agent = AgenticExecutor(config, client)
        try:
            result = agent.run(task, inputs)
            record.update(result)
        except Exception:
            # Preserve partial traces when the control loop fails.
            if not record.get("agent_trace"):
                record["agent_trace"] = agent.trace
                record["selected_chunks"] = agent.selected_chunks
            raise
        parsed = record["parsed_output"]
        _validate_output(str(task["experiment"]), parsed)
    except Exception as error:
        record["errors"].append(str(error))
    return record


def run_task_with_retries(
    task: dict[str, Any],
    config: HarnessConfig,
    client: OpenAICompatibleClient,
    max_attempts: int = 3,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    failed_attempts: list[dict[str, Any]] = []
    record: dict[str, Any] | None = None
    for attempt_index in range(max_attempts):
        attempted = run_task(task, config, client)
        if not attempted["errors"]:
            record = attempted
            break
        failed_attempts.append(
            {
                "attempt": attempt_index + 1,
                "errors": attempted["errors"],
            }
        )
        record = attempted
        if attempt_index + 1 < max_attempts:
            print(
                f"{task['task_id']} prompt={attempted['prompt_version']} "
                f"attempt={attempt_index + 1}/{max_attempts} retry"
            )

    assert record is not None
    record["max_attempts"] = max_attempts
    record["attempt_count"] = len(failed_attempts) + (0 if record["errors"] else 1)
    record["failed_attempts"] = failed_attempts
    return record


def run_tasks(
    tasks: Iterable[dict[str, Any]],
    config: HarnessConfig,
    output_path: Path,
    max_attempts: int = 3,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    client = OpenAICompatibleClient(config.generator)
    tasks = list(tasks)
    records: list[dict[str, Any]] = []

    def _run_task(task: dict[str, Any]) -> dict[str, Any]:
        return run_task_with_retries(task, config, client, max_attempts)

    if concurrency <= 1:
        for task in tasks:
            record = _run_task(task)
            append_jsonl(output_path, record)
            records.append(record)
            status = "error" if record["errors"] else "ok"
            print(f"{task['task_id']} prompt={record['prompt_version']} {status}")
        return records

    # Task-level parallelism: each task runs on its own thread. The
    # client is stateless per call, so it is safe to share; only the JSONL append
    # and the shared records list need a lock.
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            record = future.result()
            with write_lock:
                append_jsonl(output_path, record)
                records.append(record)
                status = "error" if record["errors"] else "ok"
                print(f"{task['task_id']} prompt={record['prompt_version']} {status}")
    return records


def _filter_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = tasks
    if args.experiment != "all":
        selected = [task for task in selected if task["experiment"] == args.experiment]
    if args.pattern:
        selected = [task for task in selected if task["pattern_id"] == args.pattern]
    if args.task_id:
        selected = [task for task in selected if task["task_id"] == args.task_id]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def command_run(args: argparse.Namespace, config: HarnessConfig) -> int:
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    manifest = build_manifest(config)
    tasks = build_tasks(manifest, config)
    save_manifest_and_tasks(manifest, tasks, config.task_dir)
    tasks = _filter_tasks(tasks, args)
    if not tasks:
        print("No matching tasks.")
        return 1

    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.run_dir / f"outputs-{_timestamp()}.jsonl"
    )
    if args.resume and output_path.exists():
        # Key resume on (task_id, prompt_version), matching released records.
        done = {
            (record.get("task_id"), record.get("prompt_version"))
            for record in load_jsonl(output_path)
            if not record.get("errors")
        }
        if done:
            before = len(tasks)
            tasks = [
                task
                for task in tasks
                if (task["task_id"], prompt_version_for()) not in done
            ]
            print(
                f"Resume: {len(done)} task/prompt-version records already complete; "
                f"{before - len(tasks)} tasks skipped."
            )
    run_tasks(
        tasks,
        config,
        output_path,
        args.max_attempts,
        concurrency=args.concurrency,
    )
    print(f"Outputs: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper's agentic executor")
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run agentic executor tasks")
    run_parser.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    run_parser.add_argument("--pattern", help="Filter by pattern ID, e.g. P1")
    run_parser.add_argument("--task-id", help="Run one task ID")
    run_parser.add_argument("--limit", type=int, help="Limit selected task count")
    run_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum total attempts for each task before recording an error",
    )
    run_parser.add_argument("--output", help="Output JSONL path")
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel (default 1 = sequential)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed task records already present in --output and append the rest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return command_run(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
