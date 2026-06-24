from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import read_text, resolve_safe_input
from .manifest import extract_saved_excerpt


FRAMEWORK_GUIDE = """The modular framework represents data as D=(D.text,D.tab) and a pattern as P=(M,T,E).
- D.text is report text and D.tab is source structured data.
- M is a mathematical pattern definition, T is its natural-language description, and E is a set of examples.
- E.text is an exact contiguous excerpt from D.text. E.tab is observed compact tabular evidence supplied with an example; an internal table summary is not a new E.tab.
- A concrete component is known input, ? is a target to produce, _ is irrelevant or omitted, and [n] requests n outputs.
- A named component whose value is not supplied is unavailable; _ deliberately marks a component as irrelevant or not requested.
- Never use an unavailable or irrelevant component. Return only target components.

Read and decompose queries internally as follows:
1. D:(D.text,_), P:(M,_,{(?,_)[2]}) means use known M to find two exact E.text excerpts in D.text. D.tab, T, and E.tab are not used.
2. D:(D.text,_), P:(_,_,{(?,E.tab)}) means summarize the provided E.tab internally, search D.text for matching evidence, and return one exact E.text. Do not create E.tab.
3. D:(_,_), P:(?,?,{(E1.text,E1.tab),...}) means compare the provided paired examples, infer their shared rule, and return M and T.
These decompositions are reasoning guidance inside one model call, not additional tools or data."""


EXACT_EXCERPT_RULE = (
    "Every returned passage must be contiguous text copied exactly from the supplied report. "
    "A definition, criterion, table value, prompt instruction, heading copied without its evidence, "
    "vague description, summary, paraphrase, or invented wording is not a valid passage. "
    "Return only genuine report evidence; never fill a requested slot with an invalid substitute."
)


OUTPUT_SCHEMAS = {
    "1": '{"examples":[{"excerpt":"exact report text"},{"excerpt":"exact report text"},{"excerpt":"exact report text"}]}',
    "2": '{"excerpt":"exact report text"}',
    "3": '{"M":"mathematical definition","T":"natural-language description"}',
}

SETUP_A_OUTPUT_SCHEMAS = {
    **OUTPUT_SCHEMAS,
    "3": '{"mathematical_definition":"...","natural_language_description":"..."}',
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


def _strict_json_instruction(experiment: str, setup: str = "B") -> str:
    schemas = SETUP_A_OUTPUT_SCHEMAS if setup.upper() == "A" else OUTPUT_SCHEMAS
    return (
        "Return JSON only, without Markdown or commentary. Use exactly this schema: "
        + schemas[experiment]
    )


def build_setup_a_prompt(task: dict[str, Any], inputs: dict[str, Any]) -> list[dict[str, str]]:
    experiment = task["experiment"]
    if experiment == "1":
        user = f"""Find three passages in a report that satisfy a mathematical criterion. The criterion is given first:

{inputs['M']}

The report follows:

{inputs['D.text']}

{EXACT_EXCERPT_RULE} The mathematical criterion itself is not an eligible passage.

{_strict_json_instruction(experiment, 'A')}"""
    elif experiment == "2":
        user = f"""A compact table is given first. Find the single passage in the following report that best describes the same entities, metric, years or interval, and behavior.

{inputs['E.tab']}

The report follows:

{inputs['D.text']}

{EXACT_EXCERPT_RULE}

{_strict_json_instruction(experiment, 'A')}"""
    elif experiment == "3":
        rendered_pairs = []
        for index, pair in enumerate(inputs["paired_examples"], start=1):
            rendered_pairs.append(
                f"Pair {index}. The report passage is:\n{pair['E.text']}\n\n"
                f"Its compact table is:\n{pair['E.tab']}"
            )
        rendered = "\n\n".join(rendered_pairs)
        user = f"""Infer the recurring analytical pattern illustrated by the following paired report passages and compact tables. Produce a precise mathematical definition and a concise natural-language description. The rule should accept the examples while excluding superficially similar but structurally different cases.

{rendered}

{_strict_json_instruction(experiment, 'A')}"""
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
        components = f"""<known_components>
<M>
{inputs['M']}
</M>
</known_components>
<available_data>
<D_text>
{inputs['D.text']}
</D_text>
</available_data>"""
    elif experiment == "2":
        components = f"""<known_components>
<E_tab format="compact_csv">
{inputs['E.tab']}
</E_tab>
</known_components>
<available_data>
<D_text>
{inputs['D.text']}
</D_text>
</available_data>"""
    elif experiment == "3":
        examples = []
        for pair in inputs["paired_examples"]:
            examples.append(
                f"""<example id="{pair['example_id']}">
<E_text>
{pair['E.text']}
</E_text>
<E_tab format="compact_csv">
{pair['E.tab']}
</E_tab>
</example>"""
            )
        components = "<known_components>\n<paired_examples>\n" + "\n".join(examples) + (
            "\n</paired_examples>\n</known_components>"
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    user = f"""<framework_guide>
{FRAMEWORK_GUIDE}
</framework_guide>

<framework_query>
{task['framework_query']}
</framework_query>

<query_interpretation>
{_framework_interpretation(experiment)}
</query_interpretation>

{components}

<output_constraints>
{EXACT_EXCERPT_RULE if experiment in {'1', '2'} else 'Return only the requested M and T components.'}
Never infer or create newly verified E.tab when D.tab is unavailable.
{_strict_json_instruction(experiment)}
</output_constraints>"""
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
