"""Shared task-input materialization for the v2 setups.

The v1 prompt builders were removed; only the input-materialization helpers
remain, since runner_v2 and the tests rely on them to turn a task record into
the concrete generator inputs (and their safe reference paths).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_text, resolve_safe_input
from .manifest import extract_saved_excerpt


def materialize_generator_inputs(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    experiment = task["experiment"]
    if experiment == "1":
        m_path = resolve_safe_input(root, task["m_path"], {".txt"})
        report_path = resolve_safe_input(root, task["report_path"], {".txt"})
        return {"M": read_text(m_path), "D.text": read_text(report_path)}
    if experiment == "2":
        # Deliberately never resolves gold_e_text_path here.
        tab_path = resolve_safe_input(root, task["e_tab_path"], {".csv"})
        report_path = resolve_safe_input(root, task["report_path"], {".txt"})
        return {"E.tab": read_text(tab_path), "D.text": read_text(report_path)}
    if experiment == "3":
        pairs: list[dict[str, str]] = []
        for example in task["induction_examples"]:
            text_path = resolve_safe_input(root, example["e_text_path"], {".txt"})
            tab_path = resolve_safe_input(root, example["e_tab_path"], {".csv"})
            pairs.append(
                {
                    "example_id": example["example_id"],
                    "E.text": extract_saved_excerpt(text_path),
                    "E.tab": read_text(tab_path),
                }
            )
        return {"paired_examples": pairs}
    raise ValueError(f"Unknown experiment: {experiment}")


def generator_input_refs(task: dict[str, Any]) -> dict[str, Any]:
    experiment = task["experiment"]
    if experiment == "1":
        return {"M": task["m_path"], "D.text": task["report_path"]}
    if experiment == "2":
        return {"E.tab": task["e_tab_path"], "D.text": task["report_path"]}
    if experiment == "3":
        return {
            "paired_examples": [
                {
                    "example_id": example["example_id"],
                    "E.text": example["e_text_path"],
                    "E.tab": example["e_tab_path"],
                }
                for example in task["induction_examples"]
            ]
        }
    raise ValueError(f"Unknown experiment: {experiment}")
