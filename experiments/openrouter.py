from __future__ import annotations

import argparse
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import EndpointConfig, load_config
from .manifest import build_manifest, build_tasks, save_manifest_and_tasks
from .runner import run_tasks


DENSE_JSON_MODELS = {
    "qwen32b": "qwen/qwen3-32b",
    "llama70b": "meta-llama/llama-3.3-70b-instruct",
    "qwen72b": "qwen/qwen-2.5-72b-instruct",
    "hermes70b": "nousresearch/hermes-4-70b",
}


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


def _openrouter_generator(model: str, template: EndpointConfig) -> EndpointConfig:
    return replace(
        template,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        api_key_env="OPENROUTER_API_KEY",
        response_format={"type": "json_object"},
        extra_headers={
            "HTTP-Referer": "https://github.com/local/wipo-framework-experiments",
            "X-Title": "WIPO Framework Experiments",
        },
        retry_attempts=max(template.retry_attempts, 5),
        retry_initial_sleep_seconds=max(template.retry_initial_sleep_seconds, 2.0),
        retry_backoff_factor=max(template.retry_backoff_factor, 2.0),
        timeout_seconds=max(template.timeout_seconds, 300),
    )


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = build_manifest(config)
    all_tasks = build_tasks(manifest, config)
    save_manifest_and_tasks(manifest, all_tasks, config.output_dir)
    tasks = _filter_tasks(all_tasks, args)
    if not tasks:
        print("No matching tasks.")
        return 1

    model_ids = []
    for requested in args.model:
        model_ids.append(DENSE_JSON_MODELS.get(requested, requested))
    setups = ["A", "B", "C"] if args.setup == "all" else [args.setup.upper()]

    output_dir = Path(args.output_dir).resolve() if args.output_dir else config.output_dir / "runs"
    for model_id in model_ids:
        run_config = replace(config, generator=_openrouter_generator(model_id, config.generator))
        output_path = output_dir / f"results-{_slug(model_id)}-{_timestamp()}.jsonl"
        print(f"OpenRouter model: {model_id}")
        run_tasks(tasks, setups, run_config, output_path)
        print(f"Results: {output_path}")
    return 0


def command_list(_args: argparse.Namespace) -> int:
    for alias, model in DENSE_JSON_MODELS.items():
        print(f"{alias}\t{model}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run WIPO experiments against OpenRouter dense JSON-mode models"
    )
    parser.add_argument(
        "--config",
        default="experiments/config.openrouter-dense-llama70b.json",
        help="Base config; judge and harness settings are reused",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-models", help="List curated model aliases")
    list_parser.set_defaults(func=command_list)

    run_parser = subparsers.add_parser("run", help="Run one or more OpenRouter models")
    run_parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model alias or OpenRouter model id; may be repeated",
    )
    run_parser.add_argument("--setup", choices=["A", "B", "C", "all"], default="all")
    run_parser.add_argument("--experiment", choices=["1", "2", "3", "all"], default="all")
    run_parser.add_argument("--pattern", help="Filter by pattern ID, e.g. P1")
    run_parser.add_argument("--task-id", help="Run one task ID")
    run_parser.add_argument("--limit", type=int, help="Limit selected task count")
    run_parser.add_argument("--output-dir", help="Directory for result JSONL files")
    run_parser.set_defaults(func=command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and not args.model:
        args.model = ["llama70b"]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
