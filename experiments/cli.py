from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import HarnessConfig, load_config
from .judge import evaluate_results, summarize_evaluations
from .manifest import build_manifest, build_tasks, manifest_counts, save_manifest_and_tasks


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_corpus(config: HarnessConfig):
    manifest = build_manifest(config)
    tasks = build_tasks(manifest, config)
    return manifest, tasks


def command_manifest(args: argparse.Namespace, config: HarnessConfig) -> int:
    manifest, tasks = _load_corpus(config)
    output_dir = Path(args.output).resolve() if args.output else config.output_dir
    manifest_path, tasks_path = save_manifest_and_tasks(manifest, tasks, output_dir)
    print(json.dumps(manifest_counts(manifest, tasks), indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Tasks: {tasks_path}")
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
    parser = argparse.ArgumentParser(description="WIPO modular-framework experiment harness (v2)")
    parser.add_argument("--config", help="Path to JSON configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Build corpus manifest and tasks")
    manifest_parser.add_argument("--output", help="Output directory")

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
        "judge": command_judge,
        "summarize": command_summarize,
    }
    return commands[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
