from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .agent_v2 import FrameworkAgentV2
from .agent_v2_full import FrameworkAgentV2FullContext
from .client import OpenAICompatibleClient
from .config import HarnessConfig, load_config
from .io_utils import append_jsonl, extract_json, load_jsonl
from .manifest import build_manifest, build_tasks, save_manifest_and_tasks
from .prompts import generator_input_refs, materialize_generator_inputs
from .prompts_v2 import (
    build_setup_a_prompt_v2,
    build_setup_b_prompt_v2,
    build_setup_b_prompt_v2_query_only,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def prompt_version_for(setup: str, b_variant: str, c_variant: str) -> str:
    """Canonical prompt-version tag for a (setup, variant) combination.

    This is the single source of truth used both to stamp result records and to
    key resume: it must uniquely identify each runnable configuration. The two
    Setup B variants and the two Setup C variants share a setup label but differ
    here (``v2-agentic`` vs ``v2-agentic-full``), so resuming keys on this rather
    than on ``setup`` to keep map-reduce and full-context runs distinct.
    """
    setup = setup.upper()
    if setup == "A":
        return "v2"
    if setup == "B":
        return f"v2-{b_variant}"
    if setup == "C":
        return "v2-agentic" if c_variant == "map-reduce" else "v2-agentic-full"
    return "v2"


def _estimated_tokens(messages: list[dict[str, str]], chars_per_token: float) -> int:
    chars = sum(len(message["content"]) for message in messages)
    return int(chars / chars_per_token) + 32


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


def _build_prompt_v2(
    setup: str, task: dict[str, Any], inputs: dict[str, Any], b_variant: str
) -> list[dict[str, str]]:
    if setup == "A":
        return build_setup_a_prompt_v2(task, inputs)
    if setup == "B":
        if b_variant == "resolved":
            return build_setup_b_prompt_v2(task, inputs)
        return build_setup_b_prompt_v2_query_only(task, inputs)
    raise ValueError(f"V2 runner builds single prompts only for setup A or B, not {setup}")


def _validate_output_v2(experiment: str, parsed: dict[str, Any]) -> None:
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


def run_task_v2(
    task: dict[str, Any],
    setup: str,
    config: HarnessConfig,
    client: OpenAICompatibleClient,
    b_variant: str = "query-only",
    c_variant: str = "map-reduce",
) -> dict[str, Any]:
    setup = setup.upper()
    prompt_version = prompt_version_for(setup, b_variant, c_variant)
    record = _base_record(task, setup, prompt_version)
    try:
        inputs = materialize_generator_inputs(config.project_root, task)
        if setup == "C":
            agent_class = (
                FrameworkAgentV2 if c_variant == "map-reduce" else FrameworkAgentV2FullContext
            )
            agent = agent_class(config, client)
            try:
                result = agent.run(task, inputs)
                record.update(result)
            except Exception:
                # Preserve the partial trace on failure (e.g. max_steps) so the
                # control loop is debuggable instead of being discarded.
                if not record.get("agent_trace"):
                    record["agent_trace"] = agent.trace
                    record["selected_chunks"] = agent.selected_chunks
                raise
            parsed = record["parsed_output"]
        elif setup in {"A", "B"}:
            messages = _build_prompt_v2(setup, task, inputs, b_variant)
            estimated = _estimated_tokens(messages, config.chars_per_token_estimate)
            reserve = config.generator.max_output_tokens
            if estimated + reserve > config.context_limit_tokens:
                raise ValueError(
                    f"Estimated prompt ({estimated} tokens) plus output reserve ({reserve}) "
                    f"exceeds context_limit_tokens={config.context_limit_tokens}; inputs were not truncated"
                )
            response = client.complete(messages)
            record["raw_output"] = response.text
            parsed = extract_json(response.text)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")
            if setup == "A" and task["experiment"] == "3":
                parsed = {
                    "M": parsed.get("mathematical_definition", ""),
                    "T": parsed.get("natural_language_description", ""),
                }
            record["parsed_output"] = parsed
            if "D.text" in inputs:
                record["selected_chunks"] = ["full-report"]
            record["usage"] = response.usage
        else:
            raise ValueError(f"Unknown setup: {setup}")
        _validate_output_v2(str(task["experiment"]), parsed)
    except Exception as error:
        record["errors"].append(str(error))
    return record


def run_task_v2_with_retries(
    task: dict[str, Any],
    setup: str,
    config: HarnessConfig,
    client: OpenAICompatibleClient,
    b_variant: str = "query-only",
    max_attempts: int = 3,
    c_variant: str = "map-reduce",
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    failed_attempts: list[dict[str, Any]] = []
    record: dict[str, Any] | None = None
    for attempt_index in range(max_attempts):
        attempted = run_task_v2(task, setup, config, client, b_variant, c_variant)
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
                f"{task['task_id']} setup={setup.upper()} "
                f"prompt={attempted['prompt_version']} "
                f"attempt={attempt_index + 1}/{max_attempts} retry"
            )

    assert record is not None
    record["max_attempts"] = max_attempts
    record["attempt_count"] = len(failed_attempts) + (0 if record["errors"] else 1)
    record["failed_attempts"] = failed_attempts
    return record


def run_tasks_v2(
    tasks: Iterable[dict[str, Any]],
    setups: Iterable[str],
    config: HarnessConfig,
    output_path: Path,
    b_variant: str = "query-only",
    max_attempts: int = 3,
    concurrency: int = 1,
    c_variant: str = "map-reduce",
) -> list[dict[str, Any]]:
    client = OpenAICompatibleClient(config.generator)
    setups = list(setups)
    pairs = [(task, setup) for task in tasks for setup in setups]
    records: list[dict[str, Any]] = []

    def _run_pair(task: dict[str, Any], setup: str) -> dict[str, Any]:
        return run_task_v2_with_retries(
            task,
            setup,
            config,
            client,
            b_variant,
            max_attempts,
            c_variant,
        )

    if concurrency <= 1:
        for task, setup in pairs:
            record = _run_pair(task, setup)
            append_jsonl(output_path, record)
            records.append(record)
            status = "error" if record["errors"] else "ok"
            print(f"{task['task_id']} setup={setup.upper()} prompt={record['prompt_version']} {status}")
        return records

    # Task-level parallelism: each (task, setup) pair runs on its own thread. The
    # client is stateless per call, so it is safe to share; only the JSONL append
    # and the shared records list need a lock.
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_run_pair, task, setup): (task, setup) for task, setup in pairs}
        for future in as_completed(futures):
            task, setup = futures[future]
            record = future.result()
            with write_lock:
                append_jsonl(output_path, record)
                records.append(record)
                status = "error" if record["errors"] else "ok"
                print(f"{task['task_id']} setup={setup.upper()} prompt={record['prompt_version']} {status}")
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
    save_manifest_and_tasks(manifest, tasks, config.output_dir)
    tasks = _filter_tasks(tasks, args)
    if not tasks:
        print("No matching tasks.")
        return 1
    setups = ["A", "B", "C"] if args.setup == "all" else [args.setup.upper()]
    default_tag = "-".join(
        sorted({prompt_version_for(setup, args.b_variant, args.c_variant) for setup in setups})
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.output_dir / "runs" / f"results-{default_tag}-{_timestamp()}.jsonl"
    )
    if args.resume and output_path.exists():
        # Key resume on (task_id, prompt_version) rather than (task_id, setup) so
        # that the two Setup C variants (map-reduce vs full-context) and the two
        # Setup B variants are never conflated when they share an output file.
        # prompt_version has always been written to records, so this stays
        # backward compatible with result files from earlier runs.
        done = {
            (record.get("task_id"), record.get("prompt_version"))
            for record in load_jsonl(output_path)
            if not record.get("errors")
        }
        if done:
            before = len(tasks)
            expected_versions = [
                prompt_version_for(setup, args.b_variant, args.c_variant) for setup in setups
            ]
            tasks = [
                task
                for task in tasks
                if not all((task["task_id"], version) in done for version in expected_versions)
            ]
            print(
                f"Resume: {len(done)} task/prompt-version records already complete; "
                f"{before - len(tasks)} tasks skipped."
            )
    run_tasks_v2(
        tasks,
        setups,
        config,
        output_path,
        args.b_variant,
        args.max_attempts,
        concurrency=args.concurrency,
        c_variant=args.c_variant,
    )
    print(f"Results: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v2 A/B prompts and v2 agentic Setup C")
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run v2 Setup A/B/C generator experiments")
    run_parser.add_argument("--setup", choices=["A", "B", "C", "all"], default="all")
    run_parser.add_argument(
        "--b-variant",
        choices=["query-only", "resolved"],
        default="query-only",
        help="Setup B v2 prompt variant; ignored for setup A",
    )
    run_parser.add_argument(
        "--c-variant",
        choices=["map-reduce", "full-context"],
        default="map-reduce",
        help=(
            "Setup C E.text-Finder variant: map-reduce searches overlapping report "
            "chunks then reduces; full-context reads the whole report in one call. "
            "Ignored for setups A and B"
        ),
    )
    run_parser.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    run_parser.add_argument("--pattern", help="Filter by pattern ID, e.g. P1")
    run_parser.add_argument("--task-id", help="Run one task ID")
    run_parser.add_argument("--limit", type=int, help="Limit selected task count")
    run_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum total attempts for each task/setup before recording an error",
    )
    run_parser.add_argument("--output", help="Result JSONL path")
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of task/setup pairs to run in parallel (default 1 = sequential)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task/setup records already present (without errors) in --output and append the rest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return command_run(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
