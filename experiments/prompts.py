from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import read_text, resolve_safe_input
from .manifest import extract_saved_excerpt


FRAMEWORK_PRIMER = """The modular framework represents data as D=(D.text,D.tab) and a pattern as P=(M,T,E). D.text is report text; D.tab is structured data. M is a mathematical definition, T is a natural-language pattern description, and E is a set of examples. Each example may contain an exact report excerpt E.text and compact tabular evidence E.tab. The operator _ marks an irrelevant or omitted component, ? marks a component to produce, and [n] requests n outputs."""


OUTPUT_SCHEMAS = {
    "1": '{"examples":[{"excerpt":"exact report text"},{"excerpt":"exact report text"},{"excerpt":"exact report text"}]}',
    "2": '{"excerpt":"exact report text"}',
    "3": '{"M":"mathematical definition","T":"natural-language description"}',
}


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


def _strict_json_instruction(experiment: str) -> str:
    return (
        "Return JSON only, without Markdown or commentary. Use exactly this schema: "
        + OUTPUT_SCHEMAS[experiment]
    )


def build_setup_a_prompt(task: dict[str, Any], inputs: dict[str, Any]) -> list[dict[str, str]]:
    experiment = task["experiment"]
    if experiment == "1":
        user = f"""Find three passages in the report that satisfy the criterion below. Copy each passage exactly as it appears in the report; do not paraphrase.

Criterion:
{inputs['M']}

Report:
{inputs['D.text']}

{_strict_json_instruction(experiment)}"""
    elif experiment == "2":
        user = f"""The CSV below summarizes evidence discussed in the report. Find the single passage that best describes the same entities, metric, years or interval, and behavior. Copy the passage exactly; do not paraphrase.

CSV:
{inputs['E.tab']}

Report:
{inputs['D.text']}

{_strict_json_instruction(experiment)}"""
    elif experiment == "3":
        rendered = json.dumps(inputs["paired_examples"], ensure_ascii=False, indent=2)
        user = f"""Infer the recurring analytical pattern illustrated by these paired report passages and CSV tables. Produce a precise mathematical definition and a concise natural-language description. The rule should accept the examples while excluding superficially similar but structurally different cases.

Examples:
{rendered}

{_strict_json_instruction(experiment)}"""
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return [
        {"role": "system", "content": "Follow the request precisely and return valid JSON only."},
        {"role": "user", "content": user},
    ]


def _framework_interpretation(experiment: str) -> str:
    if experiment == "1":
        return "Use D.text and known M. Do not use D.tab. T is not provided or required. Return three E.text examples and no E.tab."
    if experiment == "2":
        return "Use D.text and the provided compact E.tab. D.tab is unavailable. Return one exact E.text matching that evidence. Do not create or return E.tab."
    if experiment == "3":
        return "Use only the provided paired E.text/E.tab examples. External D.text and D.tab are unavailable. Produce the target M and T only."
    raise ValueError(f"Unknown experiment: {experiment}")


def build_setup_b_prompt(task: dict[str, Any], inputs: dict[str, Any]) -> list[dict[str, str]]:
    experiment = task["experiment"]
    if experiment == "1":
        components = f"KNOWN M:\n{inputs['M']}\n\nAVAILABLE D.text:\n{inputs['D.text']}"
    elif experiment == "2":
        components = (
            f"KNOWN E.tab (provided compact CSV):\n{inputs['E.tab']}\n\n"
            f"AVAILABLE D.text:\n{inputs['D.text']}"
        )
    elif experiment == "3":
        components = "KNOWN PAIRED EXAMPLES:\n" + json.dumps(
            inputs["paired_examples"], ensure_ascii=False, indent=2
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    user = f"""{FRAMEWORK_PRIMER}

FRAMEWORK QUERY:
{task['framework_query']}

INTERPRETATION:
{_framework_interpretation(experiment)}

{components}

Exact excerpts must be copied verbatim from D.text. Never infer or create newly verified E.tab when D.tab is unavailable.

{_strict_json_instruction(experiment)}"""
    return [
        {
            "role": "system",
            "content": "Execute the typed framework query exactly. Return only the requested target components as valid JSON.",
        },
        {"role": "user", "content": user},
    ]


def build_prompt(setup: str, task: dict[str, Any], inputs: dict[str, Any]) -> list[dict[str, str]]:
    setup = setup.upper()
    if setup == "A":
        return build_setup_a_prompt(task, inputs)
    if setup == "B":
        return build_setup_b_prompt(task, inputs)
    raise ValueError(f"Single-shot prompt requested for unsupported setup {setup}")
