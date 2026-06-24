from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .agent import FrameworkAgent
from .client import OpenAICompatibleClient
from .config import HarnessConfig
from .io_utils import append_jsonl, extract_json
from .prompts import build_prompt, generator_input_refs, materialize_generator_inputs


def _estimated_tokens(messages: list[dict[str, str]], chars_per_token: float) -> int:
    chars = sum(len(message["content"]) for message in messages)
    return int(chars / chars_per_token) + 32


def _base_record(task: dict[str, Any], setup: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "experiment": task["experiment"],
        "setup": setup,
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


def run_task(
    task: dict[str, Any],
    setup: str,
    config: HarnessConfig,
    client: OpenAICompatibleClient,
) -> dict[str, Any]:
    setup = setup.upper()
    record = _base_record(task, setup)
    try:
        inputs = materialize_generator_inputs(config.project_root, task)
        if setup in {"A", "B"}:
            messages = build_prompt(setup, task, inputs)
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
            record["parsed_output"] = parsed
            if "D.text" in inputs:
                record["selected_chunks"] = ["full-report"]
            record["usage"] = response.usage
        elif setup == "C":
            result = FrameworkAgent(config, client).run(task, inputs)
            record.update(result)
        else:
            raise ValueError(f"Unknown setup: {setup}")
    except Exception as error:
        record["errors"].append(str(error))
    return record


def run_tasks(
    tasks: Iterable[dict[str, Any]],
    setups: Iterable[str],
    config: HarnessConfig,
    output_path: Path,
) -> list[dict[str, Any]]:
    client = OpenAICompatibleClient(config.generator)
    records: list[dict[str, Any]] = []
    for task in tasks:
        for setup in setups:
            record = run_task(task, setup, config, client)
            append_jsonl(output_path, record)
            records.append(record)
            status = "error" if record["errors"] else "ok"
            print(f"{task['task_id']} setup={setup.upper()} {status}")
    return records
