from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig, load_config
from .io_utils import append_jsonl, load_jsonl
from .manifest import build_manifest, build_tasks, save_manifest_and_tasks
from .runner import run_task


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


def run_robustness(args: argparse.Namespace, config: HarnessConfig) -> int:
    if args.samples < 2:
        raise ValueError("Robustness runs require --samples >= 2")
    if args.temperature <= 0:
        raise ValueError("Robustness runs require --temperature > 0")

    manifest = build_manifest(config)
    tasks = _filter_tasks(build_tasks(manifest, config), args)
    if not tasks:
        print("No matching tasks.")
        return 1
    save_manifest_and_tasks(manifest, build_tasks(manifest, config), config.output_dir)

    sampled_generator = replace(config.generator, temperature=args.temperature)
    sampled_config = replace(config, generator=sampled_generator)
    client = OpenAICompatibleClient(sampled_generator)
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

    run_id = f"{_slug(sampled_generator.model)}-temp{args.temperature:g}-k{args.samples}-{_timestamp()}"
    for task in tasks:
        for setup in setups:
            for sample_index in range(args.samples):
                record = run_task(task, setup, sampled_config, client)
                record.update(
                    {
                        "run_type": "stochastic_robustness",
                        "robustness_run_id": run_id,
                        "generator_model": sampled_generator.model,
                        "generator_base_url": sampled_generator.base_url,
                        "temperature": args.temperature,
                        "samples_per_condition": args.samples,
                        "sample_index": sample_index,
                        "sample_id": f"{task['task_id']}:{setup}:sample-{sample_index + 1}",
                    }
                )
                append_jsonl(output_path, record)
                status = "error" if record["errors"] else "ok"
                print(
                    f"{task['task_id']} setup={setup} "
                    f"sample={sample_index + 1}/{args.samples} {status}"
                )

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
    run_parser.add_argument("--setup", choices=["A", "B", "C", "all"], default="all")
    run_parser.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    run_parser.add_argument("--pattern", help="Filter by pattern ID, e.g. P1")
    run_parser.add_argument("--task-id", help="Run one task ID")
    run_parser.add_argument("--limit", type=int, help="Limit selected task count")
    run_parser.add_argument("--output", help="Robustness result JSONL path")

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
