from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "artifacts" / "results"
TASKS_PATH = ROOT / "experiments" / "artifacts" / "tasks.jsonl"
MANIFEST_PATH = ROOT / "experiments" / "artifacts" / "manifest.json"
OUT_PATH = Path(__file__).resolve().parent / "viewer_data.js"

MODELS = [
    ("gpt-4.1", "GPT-4.1"),
    ("gpt-oss-high", "GPT-OSS high"),
    ("gpt-oss-medium", "GPT-OSS medium"),
    ("qwen3-32B-nonreasoning", "Qwen3-32B non-reasoning"),
    ("llama-3.3-70b", "Llama3-70B"),
]

SETUPS = [
    ("A", "Pipeline A"),
    ("B-query", "B-query"),
    ("B-resolved", "B-resolved"),
    ("C", "Pipeline C"),
]

SUMMARY_SETUP_KEY = {"A": "A", "B-query": "B", "B-resolved": "B", "C": "C"}

SUMMARY_TARGETS = [
    ("Q1", "experiment_1", "E_text_alignment_scores"),
    ("Q2", "experiment_2", "E_text_alignment_score"),
    ("Q3-M", "experiment_3", "M_alignment_score"),
    ("Q3-T", "experiment_3", "T_alignment_score"),
]

EXPERIMENT_INFO = {
    "1": {
        "id": "Q1",
        "title": "Text example mining",
        "category": "Evidence retrieval",
        "plain": "Given a report and a mathematical pattern definition, return three exact report passages that instantiate the pattern.",
        "framework_query": "D:(D.text,_)\nP:(M,_,{(?,_)[3]})",
        "known": ["D.text", "M"],
        "target": ["E.text[3]"],
        "output": '{"examples":[{"excerpt":"exact report text"}]}',
        "construction": "Two pattern families are paired with three reports each, producing 6 tasks.",
        "evaluation": "Each returned excerpt must first be recovered as an exact or normalized span from the source report. Grounded excerpts are judged against the supplied M for semantic pattern alignment.",
        "score_scale": [
            "0 = not grounded in the source report",
            "1 = not an instantiation",
            "2 = weakly related but missing central conditions",
            "3 = partial or uncertain instantiation",
            "4 = valid instantiation with minor ambiguity or omitted context",
            "5 = clear and complete instantiation",
        ],
    },
    "2": {
        "id": "Q2",
        "title": "Table-guided text example recovery",
        "category": "Evidence alignment",
        "plain": "Given a report and a compact E.tab example, return the exact report passage that describes the same evidence.",
        "framework_query": "D:(D.text,_)\nP:(_,_,{(?,E.tab)})",
        "known": ["D.text", "E.tab"],
        "target": ["E.text"],
        "output": '{"excerpt":"exact report text"}',
        "construction": "Each curated paired example contributes one task: E.tab is provided and E.text is held out. Across P1 and P2 this gives 12 tasks.",
        "evaluation": "The candidate excerpt must be grounded in the report. Grounded excerpts are judged against the compact E.tab and a diagnostic paired reference passage; the reference clarifies correspondence but is not a wording template.",
        "score_scale": [
            "0 = not grounded in the source report",
            "1 = unrelated to the table evidence",
            "2 = only weakly related evidence",
            "3 = partially aligned evidence",
            "4 = same evidence with a minor omission or ambiguity",
            "5 = describes the same evidence clearly and completely",
        ],
    },
    "3": {
        "id": "Q3",
        "title": "Pattern component induction",
        "category": "Pattern induction",
        "plain": "Given four paired examples, induce the mathematical component M and the natural-language component T.",
        "framework_query": "D:(_,_)\nP:(?,?,{(E1.text,E1.tab),(E2.text,E2.tab),(E3.text,E3.tab),(E4.text,E4.tab)})",
        "stored_query_note": "The original run artifacts store this Q3 query with an ellipsis; the viewer expands it to the four concrete paired examples supplied in every Q3 task.",
        "known": ["(E.text,E.tab)^4"],
        "target": ["M", "T"],
        "output": '{"M":"mathematical definition","T":"natural-language description"}',
        "construction": "Each pattern family has six curated paired examples. Each fold supplies four pairs and holds out two, producing 3 folds per pattern and 6 tasks total.",
        "evaluation": "Generated M and T are judged separately against the canonical pattern-card components. Equivalent wording or notation is acceptable; example-specific or vague topical answers are penalized.",
        "score_scale": [
            "1 = not aligned",
            "2 = weakly related",
            "3 = captures part of the pattern",
            "4 = substantially aligned with a minor omission or ambiguity",
            "5 = equivalent clear description or operational rule",
        ],
    },
}

Q3_DISPLAY_QUERY = EXPERIMENT_INFO["3"]["framework_query"]

PIPELINE_INFO = {
    "A": {
        "title": "Pipeline A: task-engineered direct prompting",
        "purpose": "Tests whether LLMs can answer framework-derived queries when each query is manually stated as a lean natural-language task.",
        "prompt_structure": [
            "Task-specific natural-language instruction",
            "Relevant inputs only",
            "Exact output schema",
            "Rules that emphasize grounding, generalization, and JSON output",
        ],
    },
    "B-query": {
        "title": "Pipeline B-query: structured prompting without resolution",
        "purpose": "Tests whether the model can infer the task meaning from framework rules and typed query notation.",
        "prompt_structure": [
            "Framework specification",
            "Typed framework query",
            "Available inputs in labeled sections",
            "Query instruction asking the model to parse the typed query",
            "Output rules and JSON schema",
        ],
    },
    "B-resolved": {
        "title": "Pipeline B-resolved: structured prompting with natural-language resolution",
        "purpose": "Tests the same reusable structured prompt interface while reducing the burden of interpreting the typed query.",
        "prompt_structure": [
            "Framework specification",
            "Typed framework query",
            "Available inputs in labeled sections",
            "Short resolved-task block in natural language",
            "Output rules and JSON schema",
        ],
    },
    "C": {
        "title": "Pipeline C: framework-inspired agentic execution",
        "purpose": "Tests whether the framework can organize a reusable modular execution workflow with an orchestrator, component specialists, intermediate components, and a shared blackboard.",
        "prompt_structure": [
            "Orchestrator receives framework rules, typed query, parsed query summary, input handles, and blackboard",
            "Specialists correspond to M, T, and E.text",
            "Intermediate components such as M_hat and T_hat can support later steps",
            "Final answer contains only slots marked with ? in the original query",
        ],
    },
}

PROJECT_INFO = {
    "title": "LLM-Assisted Pattern Mining Through a Typed Framework",
    "subtitle": "A project viewer for framework-derived pattern-mining queries, execution pipelines, robustness runs, and judge evaluations.",
    "summary": (
        "This project studies whether modern LLMs can help with recurring pattern-mining "
        "tasks when those tasks are expressed through a typed framework. The framework "
        "represents data as D=(D.text,D.tab) and patterns as P=(M,T,E), where M is a "
        "mathematical definition, T is a natural-language description, and E is a set of "
        "paired textual/tabular examples. The experiments evaluate three framework-derived "
        "queries across two pattern families, five model settings, and four execution "
        "pipelines."
    ),
    "takeaway": (
        "Use this viewer from top to bottom if you are new to the project: start with the "
        "framework, inspect the pattern cards and task construction, then drill into prompts, "
        "outputs, scores, and evaluation details."
    ),
}

FRAMEWORK_INFO = {
    "data": {
        "D.text": "Textual data such as WIPO patent landscape reports.",
        "D.tab": "Tabular data such as compact CSV evidence or larger tables.",
    },
    "pattern": {
        "M": "Mathematical definition: an operational rule for when the pattern occurs.",
        "T": "Textual description: a natural-language summary of the pattern behavior.",
        "E": "Examples: a set of paired examples e_i=(E_i.text,E_i.tab).",
        "E.text": "A report passage that instantiates the pattern.",
        "E.tab": "A compact tabular instance corresponding to the textual example.",
    },
    "markers": [
        {"symbol": "named slot", "meaning": "provided input, such as M, T, D.text, or E.tab"},
        {"symbol": "?", "meaning": "target component to produce"},
        {"symbol": "_", "meaning": "not provided and not requested as a final output"},
        {"symbol": "[n]", "meaning": "fixed number of target outputs requested"},
    ],
    "notes": [
        "A typed query describes the task interface; it does not prescribe one execution method.",
        "A final answer should contain only slots marked with ? in the original query.",
        "A component marked _ may still be derived internally as M_hat or T_hat by Pipeline C when it helps solve a target.",
    ],
    "query_catalog": [
        {
            "id": "Q1",
            "title": "Text example mining",
            "query": EXPERIMENT_INFO["1"]["framework_query"],
            "category": "Evidence retrieval",
            "description": EXPERIMENT_INFO["1"]["plain"],
            "evaluated": True,
        },
        {
            "id": "Q2",
            "title": "Table-guided text example recovery",
            "query": EXPERIMENT_INFO["2"]["framework_query"],
            "category": "Evidence alignment",
            "description": EXPERIMENT_INFO["2"]["plain"],
            "evaluated": True,
        },
        {
            "id": "Q3",
            "title": "Pattern component induction",
            "query": EXPERIMENT_INFO["3"]["framework_query"],
            "category": "Pattern induction",
            "description": EXPERIMENT_INFO["3"]["plain"],
            "evaluated": True,
        },
        {
            "id": "Q4",
            "title": "Tabular evidence retrieval",
            "query": "D:(_,D.tab)\nP:(M,_,{(_,?)[n]})",
            "category": "Tabular retrieval",
            "description": "Given tabular data and a mathematical definition, recover n tabular examples that satisfy the pattern.",
            "evaluated": False,
        },
        {
            "id": "Q5",
            "title": "Cross-modal evidence mining",
            "query": "D:(D.text,D.tab)\nP:(M,T,{(?,?)[n]})",
            "category": "Cross-modal mining",
            "description": "Given reports, tabular data, and a specified pattern, recover paired textual and tabular examples.",
            "evaluated": False,
        },
        {
            "id": "Q6",
            "title": "Example-seeded pattern mining",
            "query": "D:(D.text,D.tab)\nP:(_,_,{(E1.text,E1.tab),(?,?)[3]})",
            "category": "Example-seeded mining",
            "description": "Given one paired example, recover additional paired examples of the same pattern from text and tabular data.",
            "evaluated": False,
        },
        {
            "id": "Q7",
            "title": "Pattern formalization",
            "query": "D:(_,_)\nP:(?,T,{(E.text,E.tab)})",
            "category": "Formalization",
            "description": "Given a textual description and example pair, synthesize a mathematical definition consistent with both.",
            "evaluated": False,
        },
        {
            "id": "Q8",
            "title": "Inter-pattern transfer",
            "query": "D:(_,_)\nP1:(M1,T1,_)\nP2:(?,T2,_)",
            "category": "Transfer",
            "description": "Use one fully specified reference pattern and a second pattern description to synthesize the second pattern's mathematical definition.",
            "evaluated": False,
        },
    ],
}

PROMPT_SKELETONS = {
    "A": {
        "1": """SYSTEM: Follow the request precisely and return valid JSON only.

USER:
REPORT:
{D.text}

MATHEMATICAL PATTERN DEFINITION:
{M}

TASK:
Return exactly three passages from REPORT that instantiate the mathematical pattern definition.

RULES:
- Copy each passage exactly from REPORT.
- Do not paraphrase.
- Always return exactly three objects in examples.

OUTPUT:
{"examples":[{"excerpt":"exact report text"}, ...]}""",
        "2": """SYSTEM: Follow the request precisely and return valid JSON only.

USER:
REPORT:
{D.text}

COMPACT TABLE:
{E.tab}

TASK:
Return exactly one passage from REPORT that describes the same evidence as COMPACT TABLE.

RULES:
- Copy the passage exactly from REPORT.
- Match entities, metric, interval/years, values/trend, and pattern behavior where available.

OUTPUT:
{"excerpt":"exact report text"}""",
        "3": """SYSTEM: Follow the request precisely and return valid JSON only.

USER:
PAIRED EXAMPLES:
Example 1: {E1.text, E1.tab}
...
Example 4: {E4.text, E4.tab}

TASK:
Infer the recurring pattern shared by the paired examples.

RETURN:
- mathematical_definition
- natural_language_description

OUTPUT:
{"mathematical_definition":"...","natural_language_description":"..."}""",
    },
    "B-query": {
        "1": """SYSTEM: Execute the typed framework query exactly. Return only valid JSON.

USER:
<D_text>{D.text}</D_text>
<M>{M}</M>
<framework_spec>{D=(D.text,D.tab), P=(M,T,E), markers}</framework_spec>
<framework_query>D:(D.text,_) P:(M,_,{(?,_)[3]})</framework_query>
<query_instruction>
Parse the query using the framework specification, determine the target component(s), and return only those target component(s).
</query_instruction>
<output_rules>Copy each returned passage exactly from D.text.</output_rules>""",
        "2": """SYSTEM: Execute the typed framework query exactly. Return only valid JSON.

USER:
<D_text>{D.text}</D_text>
<E_tab>{E.tab}</E_tab>
<framework_spec>{D=(D.text,D.tab), P=(M,T,E), markers}</framework_spec>
<framework_query>D:(D.text,_) P:(_,_,{(?,E.tab)})</framework_query>
<query_instruction>
Parse the query using the framework specification, determine the target component(s), and return only those target component(s).
</query_instruction>
<output_rules>Return the exact D.text passage paired with E.tab.</output_rules>""",
        "3": """SYSTEM: Execute the typed framework query exactly. Return only valid JSON.

USER:
<paired_examples>{four E.text/E.tab pairs}</paired_examples>
<framework_spec>{D=(D.text,D.tab), P=(M,T,E), markers}</framework_spec>
<framework_query>D:(_,_) P:(?,?,{(E1.text,E1.tab),(E2.text,E2.tab),(E3.text,E3.tab),(E4.text,E4.tab)})</framework_query>
<query_instruction>
Parse the query using the framework specification, determine the target component(s), and return only those target component(s).
</query_instruction>
<output_rules>Generalize across examples and return only M and T.</output_rules>""",
    },
    "B-resolved": {
        "1": """Same reusable structured prompt as B-query, plus:

<resolved_task>
Known: D.text, M
Unavailable/not requested: D.tab, T, E.tab
Target: three E.text passages
Return exactly three report passages that instantiate M.
</resolved_task>""",
        "2": """Same reusable structured prompt as B-query, plus:

<resolved_task>
Known: D.text, E.tab
Unavailable/not requested: D.tab, M, T
Target: one E.text passage paired with the provided E.tab
Return exactly one report passage that describes the same evidence as E.tab.
</resolved_task>""",
        "3": """Same reusable structured prompt as B-query, plus:

<resolved_task>
Known: paired examples E={(E.text_i,E.tab_i)}
Unavailable/not requested: D.text, D.tab
Targets: M and T
Infer the shared recurring pattern across the paired examples.
</resolved_task>""",
    },
    "C": {
        "1": """The orchestrator receives the framework rules, parsed Q1, material inventory, and blackboard.
It may call T-Inducer to create T_hat from M, then call E.text-Finder to retrieve three report spans.
The final answer is normalized from candidate assignments for P.E[1..3].E.text.""",
        "2": """The orchestrator receives parsed Q2 and input handles for D.text and E.tab.
It may call M-Inducer or T-Inducer to create intermediate guidance from E.tab, then call E.text-Finder.
The final answer is normalized from the target E.text assignment.""",
        "3": """The orchestrator receives parsed Q3 and input handles for four paired examples.
It calls M-Inducer and/or T-Inducer to produce final M and T candidate assignments.
The final answer contains only the original ? slots: P.M and P.T.""",
    },
}

AGENT_INFO = {
    "FrameworkAgent": {
        "title": "Orchestrator",
        "role": "Parses the typed query, inspects available material handles and blackboard state, chooses the next specialist call, and returns final target assignments.",
        "reads": ["framework rules", "typed query", "parsed query summary", "material inventory", "blackboard"],
        "writes": ["specialist task packets", "final assignments"],
        "prompt_skeleton": """SYSTEM: You are the Framework Agent.
Interpret the framework query, maintain the framework state, choose one specialist agent at a time, and return final assignments only when every ? slot is resolved.

USER:
Runtime context:
{
  "framework_query": "...",
  "parsed_query": {"variable_slots": [], "concrete_slots": [], "not_provided_slots": []},
  "material_inventory": {...},
  "agent_catalog": [...],
  "blackboard": {...}
}

Return JSON action: invoke_agent or return_final.""",
    },
    "M-Inducer": {
        "title": "M specialist",
        "role": "Produces a mathematical pattern component M or an intermediate M_hat from supplied examples, textual evidence, tabular evidence, T, or T_hat.",
        "reads": ["E.text", "E.tab", "T", "T_hat", "material summaries"],
        "writes": ["M", "M_hat"],
        "prompt_skeleton": """SYSTEM: You are the M-Inducer agent.
Infer only M or M_hat from the supplied task packet and materials. Generalize across examples and avoid unsupported thresholds.

USER:
{"task_packet": {...}, "materials": {...}, "blackboard": {...}}

Return JSON with outputs: {"M_hat":"..."} or {"M":"..."}.""",
    },
    "T-Inducer": {
        "title": "T specialist",
        "role": "Produces a natural-language pattern component T or an intermediate T_hat from M, M_hat, examples, or tabular/textual evidence.",
        "reads": ["M", "M_hat", "E.text", "E.tab", "material summaries"],
        "writes": ["T", "T_hat"],
        "prompt_skeleton": """SYSTEM: You are the T-Inducer agent.
Infer only T or T_hat from the supplied task packet and materials. Keep it pattern-level, concise, and aligned with supplied evidence.

USER:
{"task_packet": {...}, "materials": {...}, "blackboard": {...}}

Return JSON with outputs: {"T_hat":"..."} or {"T":"..."}.""",
    },
    "E.text-Finder": {
        "title": "E.text specialist",
        "role": "Searches report chunks for exact contiguous spans that match the target pattern or provided E.tab evidence.",
        "reads": ["D.text chunks", "M", "T", "M_hat", "T_hat", "E.tab"],
        "writes": ["candidate E.text assignments"],
        "prompt_skeleton": """SYSTEM: You are the E.text-Finder agent.
Inspect one report chunk at a time. Return best-matching exact verbatim spans for requested E.text slots.

USER:
{"task_packet": {...}, "materials": {"D.text":"chunk text", ...}, "blackboard": {...}, "extra": {"chunk_id":"..."}}

Return JSON candidates keyed by canonical target slot.""",
    },
    "E.text-Reducer": {
        "title": "Reducer",
        "role": "Ranks the pool of chunk-level candidate excerpts and selects the top K exact spans without editing text.",
        "reads": ["candidate pool", "M/T/E.tab guidance", "blackboard"],
        "writes": ["selected excerpts"],
        "prompt_skeleton": """SYSTEM: You are the E.text Selector.
Rank a numbered pool of candidate excerpts and return the K best pool_ids. Select by pool number only; do not edit text.""",
    },
}

JUDGE_INFO = {
    "system": "You are an evaluation judge blind to the generation setup. Reason internally, apply the alignment rubric strictly, and output only the requested JSON score.",
    "blind_fields": [
        "execution pipeline",
        "generator model name",
        "prompt variant",
        "agent trace",
    ],
    "stored_output_note": "The persisted evaluation files store scores and metadata, not full judge rationales. The viewer shows the judge prompt skeleton and the stored score record.",
    "prompts": {
        "1": """Score how well this exact report excerpt instantiates the supplied mathematical pattern definition.

Mathematical pattern definition:
{canonical M}

Candidate report excerpt:
{grounded candidate E.text}

Use this alignment scale: 5 = clear and complete instantiation; 4 = valid instantiation with minor ambiguity or omitted context; 3 = partial or uncertain instantiation; 2 = weakly related but missing central conditions; 1 = not an instantiation.

Return only {"score": integer}.""",
        "2": """Score how well the candidate exact report excerpt describes the same evidence as the provided compact table.

Provided compact E.tab:
{E.tab}

Paired reference E.text:
{diagnostic reference passage}

Candidate exact report excerpt:
{grounded candidate E.text}

Use this alignment scale: 5 = describes the same evidence clearly and completely; 4 = same evidence with a minor omission or ambiguity; 3 = partially aligned evidence; 2 = only weakly related evidence; 1 = unrelated to the table evidence.

Return only {"score": integer}.""",
        "3": """Q3 uses two independent judge calls.

M call:
Compare Generated M against Canonical M using the 1-5 operational-alignment scale.

T call:
Compare Generated T against Canonical T using the 1-5 semantic-alignment scale.

Return only {"score": integer} for each call.""",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if limit is not None and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def saved_excerpt(path: Path) -> str:
    text = read_text(path)
    parts = text.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else text


def compact_string(value: Any, limit: int = 1800) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\r\n", "\n").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + f"... [truncated {len(normalized) - limit} chars]"


def compact_json(value: Any, string_limit: int = 1800, list_limit: int = 8, depth: int = 0) -> Any:
    if depth > 6:
        return "[max depth reached]"
    if isinstance(value, str):
        return compact_string(value, string_limit)
    if isinstance(value, list):
        items = [compact_json(item, string_limit, list_limit, depth + 1) for item in value[:list_limit]]
        if len(value) > list_limit:
            items.append(f"... {len(value) - list_limit} more")
        return items
    if isinstance(value, dict):
        return {
            str(key): compact_json(item, string_limit, list_limit, depth + 1)
            for key, item in value.items()
            if key not in {"raw_output"}
        }
    return value


def compact_trace(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for index, step in enumerate(trace):
        output = step.get("output", {})
        entry: dict[str, Any] = {
            "index": index,
            "action": step.get("action"),
            "input_refs": step.get("input_refs", []),
        }
        if isinstance(output, dict):
            if "action" in output:
                action = output.get("action", {})
                task_packet = action.get("task_packet", {}) if isinstance(action, dict) else {}
                entry["decision"] = {
                    "action": action.get("action"),
                    "agent": action.get("agent"),
                    "reason": compact_string(action.get("reason", ""), 900),
                    "goal": compact_string(task_packet.get("goal", ""), 600),
                    "materials_needed": task_packet.get("materials_needed", []),
                    "working_memory_inputs": task_packet.get("working_memory_inputs", []),
                    "final_assignments": compact_json(action.get("final_assignments", {}), 1200),
                }
            elif "parsed" in output:
                parsed = output.get("parsed", {})
                entry["agent_output"] = compact_json(parsed, 1000, 5)
            else:
                entry["output"] = compact_json(output, 900, 5)
        else:
            entry["output"] = compact_string(str(output), 900)
        compacted.append(entry)
    return compacted


def task_title(task: dict[str, Any]) -> str:
    exp = str(task.get("experiment"))
    pattern = task.get("pattern_id", "")
    if exp == "1":
        return f"{pattern} in {task.get('report_id', 'report')}"
    if exp == "2":
        return f"{task.get('source_example_id', task.get('task_id'))} text recovery"
    if exp == "3":
        return f"{pattern} fold {task.get('fold')}"
    return task.get("task_id", "")


def enrich_example(example: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(example)
    if "e_text_path" in example:
        enriched["E_text"] = saved_excerpt(ROOT / example["e_text_path"])
    if "e_tab_path" in example:
        enriched["E_tab"] = read_text(ROOT / example["e_tab_path"], 3500)
    return enriched


def enrich_task(task: dict[str, Any]) -> dict[str, Any]:
    enriched = {
        key: value
        for key, value in task.items()
        if key
        not in {
            "report_path",
            "m_path",
            "t_path",
        }
    }
    enriched["title"] = task_title(task)
    enriched["experiment_label"] = EXPERIMENT_INFO[str(task["experiment"])]["id"]
    enriched["stored_framework_query"] = task.get("framework_query", "")
    if str(task.get("experiment")) == "3":
        enriched["display_framework_query"] = Q3_DISPLAY_QUERY
    else:
        enriched["display_framework_query"] = task.get("framework_query", "")
    if "m_path" in task:
        enriched["M"] = read_text(ROOT / task["m_path"], 4000)
    if "t_path" in task:
        enriched["T"] = read_text(ROOT / task["t_path"], 2500)
    if "e_tab_path" in task:
        enriched["E_tab"] = read_text(ROOT / task["e_tab_path"], 3500)
    if "gold_e_text_path" in task:
        enriched["gold_E_text"] = saved_excerpt(ROOT / task["gold_e_text_path"])
    if "induction_examples" in task:
        enriched["induction_examples"] = [enrich_example(item) for item in task["induction_examples"]]
    if "heldout_examples" in task:
        enriched["heldout_examples"] = [enrich_example(item) for item in task["heldout_examples"]]
    return enriched


def load_patterns() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    patterns = []
    for pattern in manifest.get("patterns", []):
        patterns.append(
            {
                "pattern_id": pattern["pattern_id"],
                "M": read_text(ROOT / pattern["m_path"], 4000),
                "T": read_text(ROOT / pattern["t_path"], 2500),
                "examples": [enrich_example(example) for example in pattern.get("examples", [])],
            }
        )
    return patterns


def normalized_score(metrics: dict[str, Any]) -> float | None:
    if "E_text_alignment_scores" in metrics:
        scores = [float(item) for item in metrics["E_text_alignment_scores"]]
        return mean(scores) if scores else None
    if "E_text_alignment_score" in metrics:
        return float(metrics["E_text_alignment_score"])
    vals = []
    for name in ("M_alignment_score", "T_alignment_score"):
        if name in metrics:
            vals.append(float(metrics[name]))
    return mean(vals) if vals else None


def build_judgements(result: dict[str, Any], evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = evaluation.get("metrics", {})
    parsed = result.get("parsed_output", {}) if isinstance(result.get("parsed_output"), dict) else {}
    judgements: list[dict[str, Any]] = []
    if "E_text_alignment_scores" in metrics:
        examples = parsed.get("examples", [])
        scores = metrics.get("E_text_alignment_scores", [])
        for index, score in enumerate(scores):
            excerpt = ""
            if index < len(examples) and isinstance(examples[index], dict):
                excerpt = str(examples[index].get("excerpt", ""))
            judgements.append(
                {
                    "target": f"E.text[{index + 1}]",
                    "score": int(score),
                    "prediction": excerpt,
                }
            )
    elif "E_text_alignment_score" in metrics:
        judgements.append(
            {
                "target": "E.text",
                "score": int(metrics["E_text_alignment_score"]),
                "prediction": str(parsed.get("excerpt", "")),
            }
        )
    else:
        if "M_alignment_score" in metrics:
            judgements.append(
                {
                    "target": "M",
                    "score": int(metrics["M_alignment_score"]),
                    "prediction": str(parsed.get("M") or parsed.get("mathematical_definition") or ""),
                }
            )
        if "T_alignment_score" in metrics:
            judgements.append(
                {
                    "target": "T",
                    "score": int(metrics["T_alignment_score"]),
                    "prediction": str(parsed.get("T") or parsed.get("natural_language_description") or ""),
                }
            )
    return judgements


def eval_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("task_id")), str(record.get("sample_id") or record.get("sample_index")))


def result_key(record: dict[str, Any]) -> tuple[str, str]:
    return (str(record.get("task_id")), str(record.get("sample_id") or record.get("sample_index")))


def build_cases(tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for model_id, model_label in MODELS:
        for setup_id, setup_label in SETUPS:
            base = RESULTS_DIR / model_id / "robustness"
            result_path = base / f"{setup_id}.results.jsonl"
            eval_path = base / f"{setup_id}.eval.jsonl"
            if not result_path.exists() or not eval_path.exists():
                continue
            evaluations = {eval_key(item): item for item in read_jsonl(eval_path)}
            for result in read_jsonl(result_path):
                key = result_key(result)
                evaluation = evaluations.get(key, {})
                task_id = str(result.get("task_id"))
                experiment = str(result.get("experiment"))
                metrics = evaluation.get("metrics", {})
                score = normalized_score(metrics)
                case_id = f"{model_id}::{setup_id}::{task_id}::{result.get('sample_index', '0')}"
                cases.append(
                    {
                        "case_id": case_id,
                        "model": model_id,
                        "model_label": model_label,
                        "setup": setup_id,
                        "setup_label": setup_label,
                        "task_id": task_id,
                        "task_title": tasks.get(task_id, {}).get("title", task_id),
                        "experiment": experiment,
                        "experiment_label": EXPERIMENT_INFO[experiment]["id"],
                        "pattern": result.get("pattern_id"),
                        "sample_index": result.get("sample_index"),
                        "sample_id": result.get("sample_id"),
                        "score": score,
                        "metrics": metrics,
                        "evaluation_record": compact_json(evaluation, 1800, 8),
                        "judgements": build_judgements(result, evaluation),
                        "errors": {
                            "generation": result.get("errors", []),
                            "evaluation": evaluation.get("errors", []),
                            "failed_attempts": result.get("failed_attempts", []),
                        },
                        "framework_query": result.get("framework_query"),
                        "display_framework_query": Q3_DISPLAY_QUERY
                        if experiment == "3"
                        else result.get("framework_query"),
                        "parsed_output": compact_json(result.get("parsed_output", {}), 2600, 8),
                        "raw_output": compact_string(result.get("raw_output", ""), 5000),
                        "usage": result.get("usage", {}),
                        "created_at": result.get("created_at"),
                        "temperature": result.get("temperature"),
                        "generator_model": result.get("generator_model"),
                        "inputs_used": result.get("inputs_used", []),
                        "selected_chunks": compact_json(result.get("selected_chunks", []), 700, 8),
                        "agent_trace": compact_trace(result.get("agent_trace") or []),
                        "framework_agent_final": compact_json(result.get("framework_agent_final", {}), 1600, 6),
                        "source_files": {
                            "results": str(result_path.relative_to(ROOT)).replace("\\", "/"),
                            "eval": str(eval_path.relative_to(ROOT)).replace("\\", "/"),
                        },
                    }
                )
    return cases


def build_task_groups(tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for experiment in ("1", "2", "3"):
        exp_tasks = [task for task in tasks.values() if str(task.get("experiment")) == experiment]
        exp_tasks.sort(key=lambda task: str(task.get("task_id", "")))
        patterns = sorted({str(task.get("pattern_id")) for task in exp_tasks})
        reports = sorted({str(task.get("report_id")) for task in exp_tasks if task.get("report_id")})
        groups[experiment] = {
            "count": len(exp_tasks),
            "patterns": patterns,
            "reports": reports,
            "tasks": [
                {
                    "task_id": task.get("task_id"),
                    "title": task.get("title"),
                    "pattern": task.get("pattern_id"),
                    "report": task.get("report_id"),
                    "source_example_id": task.get("source_example_id"),
                    "fold": task.get("fold"),
                    "induction_examples": [
                        item.get("example_id") for item in task.get("induction_examples", [])
                    ],
                    "heldout_examples": [
                        item.get("example_id") for item in task.get("heldout_examples", [])
                    ],
                    "framework_query": task.get("display_framework_query") or task.get("framework_query"),
                }
                for task in exp_tasks
            ],
        }
    return groups


def build_result_tree(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tree: dict[str, dict[str, Any]] = {}
    for case in cases:
        model = case["model"]
        setup = case["setup"]
        experiment = case["experiment_label"]
        model_node = tree.setdefault(
            model,
            {
                "id": model,
                "label": case["model_label"],
                "path": f"experiments/artifacts/results/{model}/robustness",
                "count": 0,
                "setups": {},
            },
        )
        model_node["count"] += 1
        setup_node = model_node["setups"].setdefault(
            setup,
            {
                "id": setup,
                "label": case["setup_label"],
                "result_file": case["source_files"]["results"],
                "eval_file": case["source_files"]["eval"],
                "count": 0,
                "experiments": {},
            },
        )
        setup_node["count"] += 1
        exp_node = setup_node["experiments"].setdefault(
            experiment,
            {"id": experiment, "count": 0, "patterns": {}},
        )
        exp_node["count"] += 1
        pattern_node = exp_node["patterns"].setdefault(
            case["pattern"],
            {"id": case["pattern"], "count": 0, "case_ids": []},
        )
        pattern_node["count"] += 1
        pattern_node["case_ids"].append(case["case_id"])

    rendered: list[dict[str, Any]] = []
    for model_node in tree.values():
        setups = []
        for setup_node in model_node["setups"].values():
            experiments = []
            for exp_node in setup_node["experiments"].values():
                experiments.append(
                    {
                        **exp_node,
                        "patterns": list(exp_node["patterns"].values()),
                    }
                )
            setup_node = dict(setup_node)
            setup_node["experiments"] = sorted(experiments, key=lambda item: item["id"])
            setups.append(setup_node)
        model_node = dict(model_node)
        model_node["setups"] = sorted(setups, key=lambda item: item["id"])
        rendered.append(model_node)
    return sorted(rendered, key=lambda item: item["id"])


def summary_value(model_id: str, setup_id: str, pattern: str, experiment: str, metric: str) -> float | None:
    path = RESULTS_DIR / model_id / "robustness" / f"{setup_id}.summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    key = f"{experiment}/setup_{SUMMARY_SETUP_KEY[setup_id]}/pattern_{pattern}/{metric}"
    item = data.get("groups", {}).get(key)
    return None if item is None else round(float(item["mean"]), 4)


def build_summary_matrix() -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for pattern in ("P1", "P2", "Avg"):
        matrix[pattern] = {}
        for model_id, _ in MODELS:
            matrix[pattern][model_id] = {}
            for setup_id, _ in SETUPS:
                matrix[pattern][model_id][setup_id] = {}
                for target_id, experiment, metric in SUMMARY_TARGETS:
                    if pattern == "Avg":
                        p1 = summary_value(model_id, setup_id, "P1", experiment, metric)
                        p2 = summary_value(model_id, setup_id, "P2", experiment, metric)
                        value = None if p1 is None or p2 is None else round((p1 + p2) / 2, 4)
                    else:
                        value = summary_value(model_id, setup_id, pattern, experiment, metric)
                    matrix[pattern][model_id][setup_id][target_id] = value
    return matrix


def build_score_distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {str(score): 0 for score in range(0, 6)}
    for case in cases:
        for judgement in case.get("judgements", []):
            score = str(judgement.get("score"))
            if score in counts:
                counts[score] += 1
    return counts


def main() -> None:
    tasks = {task["task_id"]: enrich_task(task) for task in read_jsonl(TASKS_PATH)}
    cases = build_cases(tasks)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT_INFO,
        "framework": FRAMEWORK_INFO,
        "models": [{"id": model_id, "label": label} for model_id, label in MODELS],
        "setups": [{"id": setup_id, "label": label} for setup_id, label in SETUPS],
        "experiments": EXPERIMENT_INFO,
        "pipelines": PIPELINE_INFO,
        "prompt_skeletons": PROMPT_SKELETONS,
        "agents": AGENT_INFO,
        "judge": JUDGE_INFO,
        "patterns": load_patterns(),
        "tasks": tasks,
        "task_groups": build_task_groups(tasks),
        "summary_matrix": build_summary_matrix(),
        "score_distribution": build_score_distribution(cases),
        "result_tree": build_result_tree(cases),
        "cases": cases,
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT_PATH.write_text("window.RESULT_VIEWER_DATA = " + data + ";\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"Cases: {len(cases)}")


if __name__ == "__main__":
    main()
