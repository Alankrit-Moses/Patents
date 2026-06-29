from __future__ import annotations

from typing import Any


SETUP_V2_OUTPUT_SCHEMAS = {
    "1": '{"examples":[{"excerpt":"exact report text"},{"excerpt":"exact report text"},{"excerpt":"exact report text"}]}',
    "2": '{"excerpt":"exact report text"}',
    "3": '{"mathematical_definition":"...","natural_language_description":"..."}',
}

SETUP_B_V2_OUTPUT_SCHEMAS = {
    **SETUP_V2_OUTPUT_SCHEMAS,
    "3": '{"M":"mathematical definition","T":"natural-language description"}',
}


FRAMEWORK_SPEC_V2 = """D = (D.text, D.tab)
P = (M, T, E)
E is a set of paired examples: E = {(E.text_i, E.tab_i)}.

Component markers:
- concrete component: provided input
- ?: target to produce
- _: irrelevant, unavailable, or not requested

Validity rules:
- A valid query must contain at least one ? target.
- Produce only target components.
- Never use components marked _.
- Never invent D.tab or E.tab when tabular source data is unavailable.
- If E.text is a target, each returned E.text must be an exact contiguous span from D.text."""


def _strict_json_instruction(
    experiment: str, schemas: dict[str, str] = SETUP_V2_OUTPUT_SCHEMAS
) -> str:
    return (
        "Return JSON only, without Markdown or commentary. Use exactly this schema:\n"
        + schemas[experiment]
    )


def build_setup_a_prompt_v2(
    task: dict[str, Any], inputs: dict[str, Any]
) -> list[dict[str, str]]:
    experiment = task["experiment"]
    if experiment == "1":
        user = f"""REPORT:
{inputs['D.text']}

MATHEMATICAL PATTERN DEFINITION:
{inputs['M']}

TASK:
Return exactly three passages from REPORT that instantiate the mathematical pattern definition.

RULES:
- Copy each passage exactly from REPORT.
- Do not paraphrase.
- Do not return the mathematical definition itself.
- Do not return headings, isolated numbers, citations, or topical mentions unless the passage itself describes the pattern behavior.
- Always return exactly three objects in examples.
- If fewer than three valid passages exist, leave the missing excerpt values as empty strings.

{_strict_json_instruction(experiment)}"""
    elif experiment == "2":
        user = f"""REPORT:
{inputs['D.text']}

COMPACT TABLE:
{inputs['E.tab']}

TASK:
Return exactly one passage from REPORT that describes the same evidence as COMPACT TABLE.

RULES:
- Copy the passage exactly from REPORT.
- Do not paraphrase.
- Match the same entities, metric, interval or years, values or trend, and pattern behavior where available.
- Do not return table text, headings, isolated numbers, citations, or topical mentions unless the passage itself describes the table evidence.
- If no exact match is available, return the closest matching exact passage from REPORT.

{_strict_json_instruction(experiment)}"""
    elif experiment == "3":
        rendered_pairs = []
        for index, pair in enumerate(inputs["paired_examples"], start=1):
            rendered_pairs.append(
                f"""Example {index}
Report passage:
{pair['E.text']}

Compact table:
{pair['E.tab']}"""
            )
        rendered = "\n\n".join(rendered_pairs)
        user = f"""PAIRED EXAMPLES:
{rendered}

TASK:
Infer the recurring pattern shared by the paired examples.

Return:
- A mathematical definition of the recurring pattern.
- A natural-language description of the recurring pattern.

RULES:
- Generalize across the examples; do not describe only one example.
- The definition should capture the shared temporal/comparative behavior.
- Do not invent evidence outside the provided examples.
- Do not return report passages or compact tables.

{_strict_json_instruction(experiment)}"""
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return [
        {"role": "system", "content": "Follow the request precisely and return valid JSON only."},
        {"role": "user", "content": user},
    ]


def _paired_examples_xml(pairs: list[dict[str, str]]) -> str:
    rendered = []
    for pair in pairs:
        rendered.append(
            f"""<example id="{pair['example_id']}">
<E_text>
{pair['E.text']}
</E_text>
<E_tab format="compact_csv">
{pair['E.tab']}
</E_tab>
</example>"""
        )
    return "\n".join(rendered)


def build_setup_b_prompt_v2(
    task: dict[str, Any], inputs: dict[str, Any]
) -> list[dict[str, str]]:
    experiment = task["experiment"]
    if experiment == "1":
        user = f"""<D_text>
{inputs['D.text']}
</D_text>

<M>
{inputs['M']}
</M>

<framework_spec>
{FRAMEWORK_SPEC_V2}
</framework_spec>

<framework_query>
{task['framework_query']}
</framework_query>

<resolved_task>
Known: D.text, M
Unavailable/irrelevant: D.tab, T, E.tab
Target: three E.text passages
Return exactly three report passages that instantiate M.
</resolved_task>

<output_rules>
Copy each passage exactly from D.text.
Do not paraphrase.
Do not return M, headings alone, isolated values, citations, or topical mentions unless the passage itself instantiates the pattern.
Always return exactly three objects in examples.
If fewer than three valid passages exist, leave the missing excerpt values as empty strings.
</output_rules>

{_strict_json_instruction(experiment, SETUP_B_V2_OUTPUT_SCHEMAS)}"""
    elif experiment == "2":
        user = f"""<D_text>
{inputs['D.text']}
</D_text>

<E_tab format="compact_csv">
{inputs['E.tab']}
</E_tab>

<framework_spec>
{FRAMEWORK_SPEC_V2}
</framework_spec>

<framework_query>
{task['framework_query']}
</framework_query>

<resolved_task>
Known: D.text, E.tab
Unavailable/irrelevant: D.tab, M, T
Target: one E.text passage paired with the provided E.tab
Return exactly one report passage that describes the same evidence as E.tab.
</resolved_task>

<output_rules>
Copy the passage exactly from D.text.
Match the same entities, metric, interval/years, values/trend, and pattern behavior where available.
Do not return table text, headings alone, isolated values, citations, or topical mentions unless the passage itself describes the table evidence.
If no exact match is available, return the closest matching exact passage from D.text.
</output_rules>

{_strict_json_instruction(experiment, SETUP_B_V2_OUTPUT_SCHEMAS)}"""
    elif experiment == "3":
        framework_spec = FRAMEWORK_SPEC_V2.replace(
            "Never invent D.tab or E.tab when tabular source data is unavailable.",
            "Never invent external D.text or D.tab.",
        ) + "\n- For paired examples, infer only from the supplied E.text/E.tab pairs."
        user = f"""<paired_examples>
{_paired_examples_xml(inputs['paired_examples'])}
</paired_examples>

<framework_spec>
{framework_spec}
</framework_spec>

<framework_query>
{task['framework_query']}
</framework_query>

<resolved_task>
Known: paired examples E = {{(E.text_i, E.tab_i)}}
Unavailable/irrelevant: D.text, D.tab
Targets: M and T
Infer the shared recurring pattern across the paired examples.
</resolved_task>

<output_rules>
Generalize across examples.
Do not describe only one pair.
Do not invent external evidence.
Return only M and T.
</output_rules>

{_strict_json_instruction(experiment, SETUP_B_V2_OUTPUT_SCHEMAS)}"""
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return [
        {
            "role": "system",
            "content": "Execute the typed framework query exactly. Return only valid JSON.",
        },
        {"role": "user", "content": user},
    ]


def build_setup_b_prompt_v2_query_only(
    task: dict[str, Any], inputs: dict[str, Any]
) -> list[dict[str, str]]:
    experiment = task["experiment"]
    query_instruction = (
        "The required task is specified only by the typed framework query. "
        "Parse the query using the framework specification, determine the target component(s), "
        "and return only those target component(s)."
    )
    if experiment == "1":
        user = f"""<D_text>
{inputs['D.text']}
</D_text>

<M>
{inputs['M']}
</M>

<framework_spec>
{FRAMEWORK_SPEC_V2}
</framework_spec>

<framework_query>
{task['framework_query']}
</framework_query>

<query_instruction>
{query_instruction}
</query_instruction>

<output_rules>
Copy each returned passage exactly from D.text.
Do not paraphrase.
Do not return M, headings alone, isolated values, citations, or topical mentions unless the passage itself instantiates the pattern.
Always return exactly three objects in examples.
If fewer than three valid passages exist, leave the missing excerpt values as empty strings.
</output_rules>

{_strict_json_instruction(experiment, SETUP_B_V2_OUTPUT_SCHEMAS)}"""
    elif experiment == "2":
        user = f"""<D_text>
{inputs['D.text']}
</D_text>

<E_tab format="compact_csv">
{inputs['E.tab']}
</E_tab>

<framework_spec>
{FRAMEWORK_SPEC_V2}
</framework_spec>

<framework_query>
{task['framework_query']}
</framework_query>

<query_instruction>
{query_instruction}
</query_instruction>

<output_rules>
Copy each returned passage exactly from D.text.
Do not paraphrase.
When the query targets E.text paired with a provided E.tab, match the same entities, metric, interval/years, values/trend, and pattern behavior where available.
Do not return table text, headings alone, isolated values, citations, or topical mentions unless the passage itself describes the table evidence.
If no exact match is available, return the closest matching exact passage from D.text.
</output_rules>

{_strict_json_instruction(experiment, SETUP_B_V2_OUTPUT_SCHEMAS)}"""
    elif experiment == "3":
        framework_spec = FRAMEWORK_SPEC_V2.replace(
            "Never invent D.tab or E.tab when tabular source data is unavailable.",
            "Never invent external D.text or D.tab.",
        ) + "\n- For paired examples, infer only from the supplied E.text/E.tab pairs."
        user = f"""<paired_examples>
{_paired_examples_xml(inputs['paired_examples'])}
</paired_examples>

<framework_spec>
{framework_spec}
</framework_spec>

<framework_query>
{task['framework_query']}
</framework_query>

<query_instruction>
{query_instruction}
</query_instruction>

<output_rules>
Generalize across examples when the query targets pattern components.
Do not describe only one pair.
Do not invent external evidence.
</output_rules>

{_strict_json_instruction(experiment, SETUP_B_V2_OUTPUT_SCHEMAS)}"""
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    return [
        {
            "role": "system",
            "content": "Execute the typed framework query exactly. Return only valid JSON.",
        },
        {"role": "user", "content": user},
    ]
