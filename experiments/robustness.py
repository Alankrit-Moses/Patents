from __future__ import annotations

import argparse
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig, load_config
from .io_utils import append_jsonl, load_jsonl
from .manifest import build_manifest, build_tasks, save_manifest_and_tasks
from .runner_v2 import prompt_version_for, run_task_v2


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return value or "model"


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


def _sample_id(task_id: str, prompt_version: str, sample_index: int) -> str:
    return f"{task_id}:{prompt_version}:sample-{sample_index + 1}"


def _completed_key(
    task_id: str, prompt_version: str | None, sample_index: int
) -> tuple[str, str, int]:
    return (str(task_id), str(prompt_version), int(sample_index))


def _load_resume_state(path: Path) -> tuple[str | None, set[tuple[str, str, int]]]:
    if not path.exists():
        return None, set()
    records = load_jsonl(path)
    if not records:
        return None, set()
    run_id = records[0].get("robustness_run_id")
    # Match on (task_id, prompt_version, sample_index) so the two Setup C variants
    # (map-reduce vs full-context) and the two Setup B variants stay distinct even
    # in a shared file. These fields have always been written, so resume remains
    # backward compatible with result files from earlier runs.
    completed = {
        _completed_key(record["task_id"], record.get("prompt_version"), record["sample_index"])
        for record in records
        if record.get("task_id") is not None
        and record.get("sample_index") is not None
        and not record.get("errors")
    }
    return str(run_id) if run_id else None, completed


def run_robustness(args: argparse.Namespace, config: HarnessConfig) -> int:
    if args.samples < 2:
        raise ValueError("Robustness runs require --samples >= 2")
    if args.temperature <= 0:
        raise ValueError("Robustness runs require --temperature > 0")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")

    manifest = build_manifest(config)
    tasks = _filter_tasks(build_tasks(manifest, config), args)
    if not tasks:
        print("No matching tasks.")
        return 1
    save_manifest_and_tasks(manifest, build_tasks(manifest, config), config.output_dir)

    sampled_generator = replace(config.generator, temperature=args.temperature)
    sampled_config = replace(
        config,
        generator=sampled_generator,
        context_limit_tokens=args.context_limit_tokens or config.context_limit_tokens,
    )
    client = OpenAICompatibleClient(sampled_generator)
    # Setup C v2 is the agentic pipeline in run_task_v2; robustness supports it.
    setups = ["A", "B", "C"] if args.setup == "all" else [args.setup.upper()]
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.output_dir
        / "robustness"
        / (
            f"results-{_slug(sampled_generator.model)}-temp{args.temperature:g}"
            f"-k{args.samples}-{_timestamp()}.jsonl"
        )
    )

    resume_run_id, completed_sample_ids = _load_resume_state(output_path) if args.resume else (None, set())
    run_id = resume_run_id or (
        f"{_slug(sampled_generator.model)}-temp{args.temperature:g}-k{args.samples}-{_timestamp()}"
    )
    if completed_sample_ids:
        print(f"Resuming {output_path} with {len(completed_sample_ids)} completed samples")
    # Build the flat list of pending (task, setup, sample_index) units, skipping
    # any already completed on resume.
    pending: list[tuple[dict[str, Any], str, int]] = []
    for task in tasks:
        for setup in setups:
            prompt_version = prompt_version_for(setup, args.b_variant, args.c_variant)
            for sample_index in range(args.samples):
                key = _completed_key(task["task_id"], prompt_version, sample_index)
                if key in completed_sample_ids:
                    print(f"{_sample_id(task['task_id'], prompt_version, sample_index)} skip")
                    continue
                pending.append((task, setup, sample_index))

    def _run_unit(task: dict[str, Any], setup: str, sample_index: int) -> dict[str, Any]:
        prompt_version = prompt_version_for(setup, args.b_variant, args.c_variant)
        sample_id = _sample_id(task["task_id"], prompt_version, sample_index)
        failed_attempts: list[dict[str, Any]] = []
        record: dict[str, Any] | None = None
        for attempt_index in range(args.max_attempts):
            attempted = run_task_v2(
                task, setup, sampled_config, client, args.b_variant, args.c_variant
            )
            if not attempted["errors"]:
                record = attempted
                break
            failed_attempts.append(
                {"attempt": attempt_index + 1, "errors": attempted["errors"]}
            )
            record = attempted
            if attempt_index + 1 < args.max_attempts:
                print(
                    f"{task['task_id']} setup={setup} "
                    f"sample={sample_index + 1}/{args.samples} "
                    f"attempt={attempt_index + 1}/{args.max_attempts} retry"
                )
        assert record is not None
        record.update(
            {
                "run_type": "stochastic_robustness",
                "robustness_run_id": run_id,
                "robustness_prompt_version": "v2",
                "b_variant": args.b_variant,
                "c_variant": args.c_variant,
                "context_limit_tokens": sampled_config.context_limit_tokens,
                "generator_model": sampled_generator.model,
                "generator_base_url": sampled_generator.base_url,
                "temperature": args.temperature,
                "samples_per_condition": args.samples,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "max_attempts": args.max_attempts,
                "attempt_count": len(failed_attempts) + (0 if record["errors"] else 1),
                "failed_attempts": failed_attempts,
            }
        )
        return record

    write_lock = threading.Lock()

    def _handle(record: dict[str, Any], task: dict[str, Any], setup: str, sample_index: int) -> None:
        with write_lock:
            append_jsonl(output_path, record)
            status = "error" if record["errors"] else "ok"
            print(
                f"{task['task_id']} setup={setup} "
                f"sample={sample_index + 1}/{args.samples} {status}"
            )

    if args.concurrency <= 1:
        for task, setup, sample_index in pending:
            record = _run_unit(task, setup, sample_index)
            _handle(record, task, setup, sample_index)
    else:
        # The client is stateless per call, so sharing it across threads is safe;
        # only the JSONL append needs the lock.
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(_run_unit, task, setup, sample_index): (task, setup, sample_index)
                for task, setup, sample_index in pending
            }
            for future in as_completed(futures):
                task, setup, sample_index = futures[future]
                record = future.result()
                _handle(record, task, setup, sample_index)

    print(f"Robustness results: {output_path}")
    return 0


def _metric_values(metrics: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[key] = [float(value)]
        elif isinstance(value, list) and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            values[key] = [float(item) for item in value]
    return values


def summarize_robustness(path: Path, success_threshold: float = 4.0) -> dict[str, Any]:
    records = [record for record in load_jsonl(path) if not record.get("errors")]
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    units: dict[tuple[str, str, str, str, str, int], list[float]] = defaultdict(list)

    for record in records:
        experiment = str(record["experiment"])
        setup = str(record["setup"])
        pattern = str(record.get("pattern_id", ""))
        task_id = str(record["task_id"])
        for metric, values in _metric_values(record.get("metrics", {})).items():
            groups[(experiment, setup, pattern, metric)].extend(values)
            for position, value in enumerate(values):
                units[(experiment, setup, pattern, metric, task_id, position)].append(value)

    summary: dict[str, Any] = {
        "success_threshold": success_threshold,
        "groups": {},
    }
    for key, values in sorted(groups.items()):
        experiment, setup, pattern, metric = key
        matching_units = [
            samples
            for unit_key, samples in units.items()
            if unit_key[:4] == key
        ]
        best_values = [max(samples) for samples in matching_units]
        group_name = f"experiment_{experiment}/setup_{setup}/pattern_{pattern}/{metric}"
        payload: dict[str, Any] = {
            "observations": len(values),
            "mean": mean(values),
            "population_stddev": pstdev(values) if len(values) > 1 else 0.0,
            "best_of_k_mean": mean(best_values),
            "success_at_k": mean(
                [1.0 if best >= success_threshold else 0.0 for best in best_values]
            ),
        }
        if metric.startswith("E_text"):
            payload["exact_excerpt_retention_rate"] = mean(
                [1.0 if value > 0 else 0.0 for value in values]
            )
        summary["groups"][group_name] = payload
    return summary


def command_summarize(args: argparse.Namespace) -> int:
    summary = summarize_robustness(Path(args.evaluations).resolve(), args.success_threshold)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supplementary stochastic-robustness runs for the WIPO harness"
    )
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate repeated stochastic samples")
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument("--samples", type=int, default=3)
    run_parser.add_argument(
        "--context-limit-tokens",
        type=int,
        help="Override context_limit_tokens for prompt-size checks",
    )
    run_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum total attempts for each logical sample (default: 3)",
    )
    run_parser.add_argument("--setup", choices=["A", "B", "C", "all"], default="all")
    run_parser.add_argument(
        "--prompt-version",
        choices=["v2"],
        default="v2",
        help="Only v2 is supported (kept for backward-compatible invocation)",
    )
    run_parser.add_argument(
        "--b-variant",
        choices=["query-only", "resolved"],
        default="query-only",
        help="Setup B v2 prompt variant; ignored for setup A and C",
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
    run_parser.add_argument("--output", help="Robustness result JSONL path")
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of (task, setup, sample) units to run in parallel (default 1 = sequential)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="If output JSONL already exists, skip completed sample_ids and append only remaining runs",
    )

    summary_parser = subparsers.add_parser(
        "summarize", help="Summarize a judged robustness JSONL"
    )
    summary_parser.add_argument("--evaluations", required=True)
    summary_parser.add_argument("--success-threshold", type=float, default=4.0)
    summary_parser.add_argument("--output", help="Optional summary JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "summarize":
        return command_summarize(args)
    return run_robustness(args, load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
