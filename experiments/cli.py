from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


from .config import HarnessConfig, load_config
from .evaluation import evaluate_results
from .task_construction import build_manifest, build_tasks, manifest_counts, save_manifest_and_tasks


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_corpus(config: HarnessConfig):
    manifest = build_manifest(config)
    tasks = build_tasks(manifest, config)
    return manifest, tasks


def command_manifest(args: argparse.Namespace, config: HarnessConfig) -> int:
    manifest, tasks = _load_corpus(config)
    output_dir = Path(args.output).resolve() if args.output else config.task_dir
    manifest_path, tasks_path = save_manifest_and_tasks(manifest, tasks, output_dir)
    print(json.dumps(manifest_counts(manifest, tasks), indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Tasks: {tasks_path}")
    return 0


def command_judge(args: argparse.Namespace, config: HarnessConfig) -> int:
    outputs_path = Path(args.outputs).resolve()
    tasks_path = (
        Path(args.tasks).resolve() if args.tasks else config.task_dir / "tasks.jsonl"
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.run_dir / f"evaluations-{_timestamp()}.jsonl"
    )
    evaluate_results(outputs_path, tasks_path, output_path, config, concurrency=args.concurrency)
    print(f"Evaluations: {output_path}")
    return 0




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pattern-engineering study utilities")
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Build corpus manifest and tasks")
    manifest_parser.add_argument("--output", help="Output directory")

    judge_parser = subparsers.add_parser("judge", help="Blind-judge result JSONL")
    judge_parser.add_argument("--outputs", required=True, help="Executor output JSONL")
    judge_parser.add_argument("--tasks", help="Task JSONL; defaults to experiments/tasks/tasks.jsonl")
    judge_parser.add_argument("--output", help="Evaluation JSONL path")
    judge_parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Number of records to judge in parallel (default 12)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    commands = {
        "manifest": command_manifest,
        "judge": command_judge,
    }
    return commands[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
