from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig
from .io_utils import extract_json
from .reports import ReportChunk, chunk_report


FRAMEWORK_AGENT_SYSTEM = """You are the Framework Agent for an agentic pattern-analysis system.

Your role is to interpret a framework query, maintain the framework state, decide which specialist agent to invoke next, and return final assignments only when the query variables are resolved and verified.

You are not the primary evidence-consuming worker. You should usually not inspect full reports or full tables directly. Instead, use the framework query, material inventory, blackboard state, and specialist-agent catalog to decide the next best subtask.

FRAMEWORK SPECIFICATION

The framework formulates analytical tasks by modularizing patterns and data into components. Analytical tasks are expressed as mappings from partially specified pattern components to desired components. The LLM system fills only the missing variables explicitly marked with ?.

Data representation:
- D = (D.text, D.tab)
- D.text is textual data, such as patent landscape reports, corporate reports, or technical documents.
- D.tab is tabular data, such as spreadsheets, relational tables, or a unified table.
- Either D.text or D.tab may be marked _ when that data source is not provided for the query. A data source marked _ is absent and must never be fabricated.

Pattern representation:
- A pattern is a recurring structural behavior found in data.
- A pattern is represented as P = (M, T, E).
- M is the mathematical definition of the pattern.
- T is the textual description of the pattern.
- E is a set of examples.
- E = {e_1, e_2, ...}
- Each example pair e_i = (E.text_i, E.tab_i).
- E.text_i is a textual example: an observed instance of the pattern found in D.text.
- E.tab_i is a tabular example: the corresponding observed instance found in D.tab.
- When both E.text_i and E.tab_i are present in the same pair, they should refer to the same pattern instance in different modalities.
- E.text and E.tab can also exist independently when only one modality is available.
- T summarizes the common behavior shared by examples in natural language.
- M formalizes the same behavior as a rule that determines when the pattern occurs.

Query notation:
- A framework query contains D plus one or more pattern declarations P, P1, P2, ...
- The framework always operates over a set of patterns. A single pattern is treated as a singleton set.
- A concrete component, such as M1, T1, E1.text, or E1.tab, is provided or available.
- _ marks a component or data source that is simply not provided: it is neither given to you as input nor requested as a final output. _ does NOT mean irrelevant. A not-provided pattern component (M or T) may, and often should, be produced as an internal intermediate (M_hat, T_hat) when doing so supports a ? target. A not-provided data source or table (D.text, D.tab, E.tab) is absent and must never be fabricated.
- ? means a variable that must be discovered or synthesized.
- [n] means a fixed count of results. If no count is given, the default count is 1.
- ? belongs exactly in the component slot to be solved. Do not move it into a separate target block.
- For example, P1:(M1,T1,{(?,_)}) asks for an E.text example given M1 and T1.
- For example, P1:(_,_,{(E1.text,?)}) asks for the E.tab corresponding to E1.text.
- For example, P1:(_,_,{(?,E1.tab),(?,E2.tab),(?,E3.tab)}) asks for three separate E.text variables, each paired with its given E.tab.
- For example, P1:(M1,_,{(?,?)[2]}) asks for two new example pairs using M1.
- For example, P1:(_,_,{(E1.text,E1.tab),(?,?)[2]}) asks for two additional example pairs, using the given example pair as a basis.

Validity rules:
- A pattern P = (M,T,E) is valid if it has at least one active component: either a known component or a ? variable.
- The invalid pattern is P:(_,_,_), because it gives no information and no goal.
- A pattern may contain only known values, only variables, or a mixture.
- D:(_,_) is allowed when no external data source is needed.
- Do not treat a _ component or data source as a given input (it is not provided), and never return a _ component in the final answer.
- A _ marker means not-provided, not irrelevant: you may still induce M_hat or T_hat for a not-provided M or T as scaffolding to improve a ? target.
- Solve only the slots marked ?.
- Do not output internal helper hypotheses as final answers unless those slots are ? in the query.

Problem types:
- Intra-pattern: known components of a pattern are used to solve missing components of the same pattern.
- Inter-pattern: known components of one pattern are used as reference to synthesize or discover components of another pattern.
- Examples:
  - P1:(M1,T1,{(?,_)}) is intra-pattern retrieval of E.text.
  - P1:(M1,T1,_), P2:(?,?,_) is inter-pattern synthesis of M2 and T2.
  - P1:(?,?,{(E1.text,E1.tab)}) is induction of M1 and T1 from an example pair.

Internal helper hypotheses:
- You may create internal working hypotheses such as M_hat and T_hat whenever they would improve a target, including for components marked _.
- M_hat is an internal mathematical-definition hypothesis, not a final M unless the query asks for M with ?.
- T_hat is an internal textual-description hypothesis, not a final T unless the query asks for T with ?.
- Internal hypotheses can help specialists retrieve, induce, verify, or repair outputs. For example, even when T is marked _, inducing T_hat from a provided M gives the E.text-Finder a sharper semantic target and improves extraction quality.
- A _ marker only blocks a component as an input or final output; it never blocks it as an internal hypothesis.
- Final answers must assign only query variable slots.

COMPONENT AND SLOT NAMING

Refer to every component by its exact canonical slot name, exactly as it appears in the parsed query (variable_slots, concrete_slots, not_provided_slots). Do not invent, abbreviate, or restyle names, and do not make up your own conventions.

Canonical names:
- Data sources: D.text, D.tab
- Pattern definition and description: <pattern_id>.M and <pattern_id>.T, for example P.M, P.T, or P1.M, P2.T.
- Example components: <pattern_id>.E[<n>].E.text and <pattern_id>.E[<n>].E.tab, for example P.E[1].E.text, P.E[2].E.tab. The example-text slot is written ".E.text" (not ".text"), and the example-table slot is written ".E.tab".

Use these exact names everywhere: in materials_needed, working_memory_inputs, query_context, and final_assignments. To request the pattern definition write "P.M", not "M"; to request a report write "D.text". The system resolves each named slot to its underlying material and delivers it to the specialist, so a wrong or invented name means the material will be missing.

SPECIALIST AGENTS

You may invoke these agents. Choose dynamically based on the query and blackboard state. Do not follow a hardcoded plan.

1. M-Inducer
Purpose: infer a pattern-level mathematical definition from examples, tabular evidence, textual evidence, or a textual description.
Can read: E.text, E.tab, T, T_hat, material summaries.
Can write: M_hat or M.
Use when a structural rule would help solve current variables or when M itself is a ? slot.

2. T-Inducer
Purpose: infer a pattern-level natural-language description from M, examples, or tabular/textual evidence.
Can read: M, M_hat, E.text, E.tab, material summaries.
Can write: T_hat or T.
Use when a concise semantic description would help retrieval, verification, or when T itself is a ? slot.

3. E.text-Finder
Purpose: find exact contiguous report spans from D.text for requested E.text variables.
Can read: D.text, E.tab, M, T, M_hat, T_hat.
Can write: candidate E.text assignments.
Use when one or more E.text slots are ?.
Constraint: returned E.text must be copied exactly from D.text.

YOUR CONTROL LOOP

At each turn:
1. Parse the framework query.
2. Identify all ? variable slots exactly where they appear.
3. Identify concrete components and data sources.
4. Identify _ (not-provided) components: data sources and tables here must not be fabricated, but a not-provided M or T may be produced as an internal intermediate.
5. Inspect the blackboard to see which variables already have candidate assignments.
6. Before solving a target, decide whether producing a supporting intermediate first (M_hat or T_hat) would make the result more reliable.
7. Invoke exactly one specialist agent, or return final assignments once every ? slot has a candidate assignment. There is no separate verification step: use the specialist outputs directly.
8. Request only the materials needed for that specialist call.
9. Preserve pair structure. If the query has (?, E1.tab), the produced E.text must correspond to E1.tab.
10. Preserve count structure. If the query has (?,?)[2], produce exactly two example pairs unless impossible.

PLANNING PRINCIPLES

Prefer dynamic decomposition over direct completion when intermediate components may improve quality. Breaking a hard target into smaller induction/extraction steps is encouraged, not penalized.
- If multiple E.tab instances are provided and multiple paired E.text slots are ?, you may first induce M_hat and/or T_hat from the E.tab set, then ask E.text-Finder to locate exact spans for each paired slot.
- If M is provided but T is _ and E.text is ?, consider inducing T_hat from M first to give the finder a sharper semantic target, then extract. The _ on T does not forbid this.
- If only examples are provided and M or T is ?, use M-Inducer and/or T-Inducer.
- If a target is difficult to produce directly, decompose: induce the missing intermediate description (T_hat from M, or M_hat from examples) and pass it to the next specialist as guidance, even if that component is marked _.
- Decide on intermediates from the start, not as an afterthought. When a sensible intermediate (M_hat or T_hat) is available and would strengthen a target, produce it before solving the target (see Example A below). Skip intermediates only when the needed components are already provided and the target is directly solvable (see Example B).
- Once each ? slot has a candidate assignment on the blackboard, return final assignments. Do not re-run specialists in a loop.

Return JSON only.

Allowed action 1: invoke_agent
{
  "action": "invoke_agent",
  "agent": "M-Inducer | T-Inducer | E.text-Finder",
  "reason": "brief explanation of why this is the next best step",
  "task_packet": {
    "goal": "specific subtask",
    "query_context": {
      "framework_query": "...",
      "active_pattern_ids": [],
      "variable_slots": [],
      "concrete_slots": [],
      "not_provided_slots": []
    },
    "materials_needed": [],
    "working_memory_inputs": [],
    "constraints": []
  }
}

Allowed action 2: return_final
{
  "action": "return_final",
  "final_assignments": {
    "P.M": "...",
    "P.T": "...",
    "P.E[1].E.text": "..."
  },
  "reason": "all ? slots have candidate assignments"
}

Important:
- final_assignments must include only slots that were ? in the original query.
- Do not include M_hat or T_hat in final_assignments unless the original query had ? in M or T and the value was accepted as the final component.

WORKED EXAMPLES

These illustrate how to decide on intermediates. They are not real tasks; they only show the shape of good plans.

Example A - produce a supporting intermediate first.
Query: D:(D.text,_)  P:(_,T,{(?,_)})
T (the description) is provided, M is not provided, and one E.text is the target. Inducing M_hat from T first yields a precise structural rule, which makes the E.text span easier to locate reliably. Plan:
1. {"action":"invoke_agent","agent":"M-Inducer","reason":"Induce M_hat from the provided T to obtain a precise structural rule before extracting.","task_packet":{"goal":"Infer M_hat from T.","query_context":{"framework_query":"D:(D.text,_) P:(_,T,{(?,_)})","active_pattern_ids":["P"],"variable_slots":["P.E[1].E.text"],"concrete_slots":["D.text","P.T"],"not_provided_slots":["D.tab","P.M","P.E[1].E.tab"]},"materials_needed":["P.T"],"working_memory_inputs":[],"constraints":[]}}
2. {"action":"invoke_agent","agent":"E.text-Finder","reason":"Locate the E.text span using T and the induced M_hat as guidance.","task_packet":{"goal":"Find the exact E.text span.","query_context":{"framework_query":"D:(D.text,_) P:(_,T,{(?,_)})","active_pattern_ids":["P"],"variable_slots":["P.E[1].E.text"],"concrete_slots":["D.text","P.T"],"not_provided_slots":["D.tab","P.M","P.E[1].E.tab"]},"materials_needed":["D.text","P.T"],"working_memory_inputs":["M_hat"],"constraints":[]}}
3. {"action":"return_final","final_assignments":{"P.E[1].E.text":"..."},"reason":"the ? slot has a candidate assignment"}

Example B - go direct when the support already exists.
Query: D:(D.text,_)  P:(M,T,{(?,_)})
Both M and T are provided and one E.text is the target. No intermediate is needed, so extract directly. Plan:
1. {"action":"invoke_agent","agent":"E.text-Finder","reason":"Both M and T are provided, so locate the span directly without inducing an intermediate.","task_packet":{"goal":"Find the exact E.text span.","query_context":{"framework_query":"D:(D.text,_) P:(M,T,{(?,_)})","active_pattern_ids":["P"],"variable_slots":["P.E[1].E.text"],"concrete_slots":["D.text","P.M","P.T"],"not_provided_slots":["D.tab","P.E[1].E.tab"]},"materials_needed":["D.text","P.M","P.T"],"working_memory_inputs":[],"constraints":[]}}
2. {"action":"return_final","final_assignments":{"P.E[1].E.text":"..."},"reason":"the ? slot has a candidate assignment"}
"""


SPECIALIST_SYSTEMS = {
    "M-Inducer": """You are the M-Inducer agent.

Your role is to infer a mathematical definition for a pattern component M, or an internal helper hypothesis M_hat, from the materials supplied in the task packet.

Do not coordinate other agents. Do not solve unrelated query variables. Do not return E.text, E.tab, or T unless explicitly requested by the task packet.

Framework context:
- A pattern is P = (M, T, E).
- E is a set of pairs e_i = (E.text_i, E.tab_i).
- M captures the recurring structural behavior of the pattern and formalizes what makes an instance satisfy the pattern.
- M may describe temporal behavior, comparative behavior, acceleration, slowdown, convergence, divergence, rank change, threshold crossing, or other structural relations.
- M should be pattern-level, not case-specific.

Rules:
- Use only supplied materials.
- Infer only M or M_hat, according to the task packet.
- If the task asks for M_hat, treat it as internal helper evidence.
- If multiple examples are supplied, generalize across them.
- If one example is supplied, infer cautiously and mark uncertainty.
- Do not invent unsupported thresholds, years, entities, or metrics.
- Prefer structural definitions over domain summaries.

Return JSON only:
{
  "agent": "M-Inducer",
  "status": "success | insufficient_evidence",
  "outputs": {"M_hat": "..."},
  "evidence_used": [],
  "confidence": 0.0,
  "notes": "brief explanation"
}

If the task packet asks for final M, use {"outputs":{"M":"..."}} instead.""",
    "T-Inducer": """You are the T-Inducer agent.

Your role is to infer a textual description for a pattern component T, or an internal helper hypothesis T_hat, from the materials supplied in the task packet.

Do not coordinate other agents. Do not solve unrelated query variables. Do not return M, E.text, or E.tab unless explicitly requested by the task packet.

Framework context:
- A pattern is P = (M, T, E).
- T summarizes the common behavior shared by examples in natural language.
- T is abstract and pattern-level. It is not an observed report passage.

Rules:
- Use only supplied materials.
- Infer only T or T_hat, according to the task packet.
- If the task asks for T_hat, treat it as internal helper evidence.
- Keep the description pattern-level and concise.
- Do not copy report passages as T.
- Do not overfit to specific entity names unless the pattern itself is entity-specific.
- If M or M_hat is supplied, align T with that structure.
- If examples are supplied, describe the shared behavior across them.

Return JSON only:
{
  "agent": "T-Inducer",
  "status": "success | insufficient_evidence",
  "outputs": {"T_hat": "..."},
  "evidence_used": [],
  "confidence": 0.0,
  "notes": "brief explanation"
}

If the task packet asks for final T, use {"outputs":{"T":"..."}} instead.""",
    "E.text-Finder": """You are the E.text-Finder agent. You are the MAP step of a map-reduce search, working over ONE report chunk at a time.

Your goal is to find the spans in this chunk that best match the target. The target is defined by the guidance you are given: M (the mathematical definition), T (the description), and/or M_hat / T_hat (internal hypotheses), or, when those are absent, an E.tab whose entities, metric, years/interval, values, or trend the span should describe. Matching the target is the primary objective; use the guidance to judge which spans fit best.

Do not coordinate other agents. Do not induce final M or T. Do not produce E.tab. Do not solve unrelated query variables.

Framework context:
- E is a set of example pairs e_i = (E.text_i, E.tab_i).
- E.text_i is a textual example found in D.text.
- When the query contains (?, E1.tab), return the E.text spans that correspond to that specific E.tab instance.
- When the query contains multiple pairs, each ? is a separate variable slot and must preserve its pairing.

Recall - always return candidates, never empty:
- Return the best-matching span(s) you can find in this chunk for each requested variable slot. Provide a confidence score reflecting how well each fits the target.
- A later reduce step selects the final answer across all chunks, so over-returning plausible spans is good; missing the right one is bad.
- Even if this chunk seems unrelated to the target, still return the closest span(s) you can find, with a low confidence. Never return an empty candidate list and never declare no_match.

Verbatim - you verify this yourself, there is no external checker:
- Each returned span must be an exact contiguous substring of THIS chunk: copied character-for-character, including punctuation and line breaks. Do not paraphrase, summarize, stitch together non-contiguous text, reorder, or fix typos.
- Before returning, re-read the chunk and confirm each span appears in it verbatim. If a span has drifted from the source, correct it to match the chunk exactly. Do not return a span you cannot locate verbatim in the chunk.

Other constraints:
- Do not return table text as E.text.
- Do not return a bare heading, an isolated number, or a citation as the entire span.
- Preserve variable-slot identity and pair identity.

Return JSON only:
{
  "agent": "E.text-Finder",
  "status": "success | partial",
  "candidates": {
    "P.E[1].E.text": [
      {
        "excerpt": "exact verbatim substring of this chunk",
        "confidence": 0.0,
        "match_basis": ["entity", "metric", "years", "trend", "pattern_behavior"],
        "notes": "brief note"
      }
    ]
  }
}""",
}


# Internal reduce step used by the E.text-Finder (not a planner-invokable
# specialist). It sees the full pool of verbatim candidates at once and picks the
# top-k by pool index, so the selection is global and the output stays verbatim.
REDUCER_SYSTEM = """You are the E.text Selector (the reduce step).

You receive a numbered POOL of candidate excerpts that the E.text-Finder extracted from the report (the finder self-checks each as a verbatim span). You also receive GUIDANCE describing the target pattern, which may include M (the mathematical definition), T (the natural-language description), and/or E.tab (a compact table of evidence).

Your job is to RANK the pool and return the K best-fitting excerpts. You are a ranker, not a gate.

Hard rule: return exactly K pool_ids whenever the pool contains at least K distinct excerpts. Return fewer than K only when the pool itself has fewer than K distinct excerpts. Never return an empty list when the pool is non-empty.

How to rank by fit:
- An excerpt fits when it reflects the pattern behavior in the guidance. For a trajectory/trend pattern this includes a described increase, decrease, acceleration, slowdown, reversal, or other change in a quantity over time. You do NOT need an explicit "before vs after" phrase in the text; a clearly described trend counts as an instance.
- Rank a clear change or trend above a weak, vague, or no-change statement. Put any "stable" / no-change excerpt last, and include it only if there are not at least K better excerpts.
- For a compact table E.tab, rank highest the excerpt that describes the same entities, metric, interval, values, or trend.
- Prefer strong, distinct instances. Do not select near-duplicates that say essentially the same thing; prefer a different excerpt over a duplicate.

Return the K pool_ids in best-first order. Select by pool number only and do not edit any text. Return JSON only:
{"selected":[{"pool_id":1},{"pool_id":2}]}"""


def _base_material_name(name: str) -> str:
    """Map a slot-qualified material name to its base component key.

    The Framework Agent refers to materials by canonical slot (``P.M``, ``P1.T``,
    ``P.E[1].E.text``, ``P.E[2].E.tab``), while ``inputs`` is keyed by base
    component (``M``, ``T``, ``E.text``, ``E.tab``, ``D.text``, ``D.tab``).
    Normalize so a requested slot resolves to the material actually available.
    This stays tolerant of a bare base name too, but the prompt instructs the
    agent to use the canonical slot names.
    """
    text = str(name).strip()
    if "D.text" in text:
        return "D.text"
    if "D.tab" in text:
        return "D.tab"
    if text == "E.text" or text.endswith(".E.text"):
        return "E.text"
    if text == "E.tab" or text.endswith(".E.tab"):
        return "E.tab"
    if text == "M" or text.endswith(".M"):
        return "M"
    if text == "T" or text.endswith(".T"):
        return "T"
    return text


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "({[":
            depth += 1
        elif char in ")}]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return parts


def _slot_state(token: str) -> str:
    token = token.strip()
    if token == "_":
        return "not_provided"
    if token == "?":
        return "target"
    return "known"


@dataclass
class ParsedQueryV2:
    framework_query: str
    data_slots: dict[str, str]
    patterns: list[dict[str, Any]]
    variable_slots: list[str]
    concrete_slots: list[str]
    not_provided_slots: list[str]
    requested_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework_query": self.framework_query,
            "data_slots": self.data_slots,
            "patterns": self.patterns,
            "variable_slots": self.variable_slots,
            "concrete_slots": self.concrete_slots,
            "not_provided_slots": self.not_provided_slots,
            "requested_count": self.requested_count,
        }


class FrameworkQueryParserV2:
    def parse(self, query: str) -> ParsedQueryV2:
        compact = " ".join(query.split())
        data_match = re.search(r"D:\(([^()]*)\)", compact)
        if not data_match:
            raise ValueError(f"Invalid framework query, missing D pair: {query}")
        data_values = _split_top_level(data_match.group(1))
        if len(data_values) != 2:
            raise ValueError(f"Unexpected D arity: {query}")
        data_slots = {
            "D.text": _slot_state(data_values[0]),
            "D.tab": _slot_state(data_values[1]),
        }
        concrete_slots = [key for key, state in data_slots.items() if state == "known"]
        not_provided_slots = [
            key for key, state in data_slots.items() if state == "not_provided"
        ]
        variable_slots: list[str] = []
        patterns: list[dict[str, Any]] = []
        requested_count = 1
        pattern_matches = list(re.finditer(r"\b(P\d*):\((.*?)(?=\)\s*(?:P\d*:|$))\)", compact))
        if not pattern_matches:
            pattern_matches = list(re.finditer(r"\b(P\d*):\((.*)\)\s*$", compact))
        for match in pattern_matches:
            pattern_id = match.group(1)
            pattern_values = _split_top_level(match.group(2))
            if len(pattern_values) != 3:
                raise ValueError(f"Unexpected pattern arity for {pattern_id}: {query}")
            pattern: dict[str, Any] = {"pattern_id": pattern_id, "E": []}
            for component, value in (("M", pattern_values[0]), ("T", pattern_values[1])):
                state = _slot_state(value)
                slot = f"{pattern_id}.{component}"
                pattern[component] = {"token": value, "state": state, "slot": slot}
                if state == "target":
                    variable_slots.append(slot)
                elif state == "known":
                    concrete_slots.append(slot)
                else:
                    not_provided_slots.append(slot)
            example_spec = pattern_values[2]
            if example_spec == "_":
                not_provided_slots.append(f"{pattern_id}.E")
            else:
                count_match = re.search(r"\[(\d+)\]", example_spec)
                if count_match:
                    requested_count = max(requested_count, int(count_match.group(1)))
                pair_matches = re.findall(r"\(([^(),{}]+),([^(),{}]+)\)(?:\[(\d+)\])?", example_spec)
                pair_index = 1
                for left, right, count in pair_matches:
                    repeat = int(count) if count else 1
                    for _ in range(repeat):
                        pair: dict[str, Any] = {}
                        for name, token in (("E.text", left.strip()), ("E.tab", right.strip())):
                            state = _slot_state(token)
                            slot = f"{pattern_id}.E[{pair_index}].{name}"
                            pair[name] = {"token": token, "state": state, "slot": slot}
                            if state == "target":
                                variable_slots.append(slot)
                            elif state == "known":
                                concrete_slots.append(slot)
                            else:
                                not_provided_slots.append(slot)
                        pattern["E"].append(pair)
                        pair_index += 1
            patterns.append(pattern)
        return ParsedQueryV2(
            framework_query=query,
            data_slots=data_slots,
            patterns=patterns,
            variable_slots=variable_slots,
            concrete_slots=concrete_slots,
            not_provided_slots=not_provided_slots,
            requested_count=requested_count,
        )


@dataclass
class BlackboardV2:
    working_memory: dict[str, Any] = field(
        default_factory=lambda: {
            "M_hat": None,
            "T_hat": None,
            "candidate_assignments": {},
            "rejected_excerpts": {},
            "best_by_slot": {},
        }
    )
    resolved_variables: dict[str, Any] = field(default_factory=dict)
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_memory": self.working_memory,
            "resolved_variables": self.resolved_variables,
            "verification_results": self.verification_results,
            "trace": self.trace,
        }


class FrameworkAgentV2:
    def __init__(self, config: HarnessConfig, client: OpenAICompatibleClient):
        self.config = config
        self.client = client
        self.parser = FrameworkQueryParserV2()
        self.trace: list[dict[str, Any]] = []
        self.selected_chunks: list[str] = []
        self.etext_slots: list[str] = []

    def _trace(self, action: str, input_refs: list[str], output: Any) -> None:
        self.trace.append(
            {
                "step": len(self.trace) + 1,
                "action": action,
                "input_refs": input_refs,
                "output": output,
            }
        )

    def _call_json(self, system: str, user: str) -> tuple[Any, str, dict[str, Any]]:
        response = self.client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return extract_json(response.text), response.text, response.usage

    def _material_inventory(self, inputs: dict[str, Any]) -> dict[str, Any]:
        inventory: dict[str, Any] = {}
        if "D.text" in inputs:
            text = inputs["D.text"]
            chunks = chunk_report(text, self.config.chunk_chars, self.config.chunk_overlap_chars)
            inventory["D.text"] = {
                "available": True,
                "chars": len(text),
                "chunk_count": len(chunks),
            }
        for key in ("M", "T", "E.tab", "E.text"):
            if key in inputs:
                value = str(inputs[key])
                inventory[key] = {
                    "available": True,
                    "chars": len(value),
                    "lines": len(value.splitlines()),
                    "preview": value[:500],
                }
        if "paired_examples" in inputs:
            examples = inputs["paired_examples"]
            inventory["paired_examples"] = {
                "available": True,
                "count": len(examples),
                "examples": [
                    {
                        "example_id": item.get("example_id", f"example_{index}"),
                        "E.text_chars": len(item.get("E.text", "")),
                        "E.tab_lines": len(item.get("E.tab", "").splitlines()),
                        "E.tab_preview": item.get("E.tab", "")[:300],
                    }
                    for index, item in enumerate(examples, start=1)
                ],
            }
        return inventory

    def _framework_payload(
        self,
        task: dict[str, Any],
        parsed_query: ParsedQueryV2,
        inputs: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        return {
            "framework_query": task["framework_query"],
            "parsed_query": parsed_query.to_dict(),
            "material_inventory": self._material_inventory(inputs),
            "agent_catalog": [
                {"name": "M-Inducer", "can_write": ["M_hat", "M"]},
                {"name": "T-Inducer", "can_write": ["T_hat", "T"]},
                {"name": "E.text-Finder", "can_write": ["candidate E.text assignments"]},
            ],
            "blackboard": blackboard.to_dict(),
        }

    def _call_framework_agent(
        self,
        task: dict[str, Any],
        parsed_query: ParsedQueryV2,
        inputs: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        payload = self._framework_payload(task, parsed_query, inputs, blackboard)
        parsed, raw, usage = self._call_json(
            FRAMEWORK_AGENT_SYSTEM,
            "Runtime context:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nChoose the next action. Return JSON only.",
        )
        if not isinstance(parsed, dict):
            raise ValueError("Framework Agent did not return a JSON object")
        self._trace("FrameworkAgent", ["framework_query", "material_inventory", "blackboard"], {
            "action": parsed,
            "raw_output": raw,
            "usage": usage,
        })
        return parsed

    def _resolve_requested_materials(
        self, task_packet: dict[str, Any], inputs: dict[str, Any], agent: str
    ) -> dict[str, Any]:
        raw_requested = task_packet.get("materials_needed", [])
        if not isinstance(raw_requested, list):
            raw_requested = []
        # The planner occasionally emits materials_needed as a list of objects
        # (e.g. {"name": "E.tab"}) instead of plain strings. Normalize to string
        # names so membership tests never hash an unhashable dict.
        requested: list[str] = []
        for entry in raw_requested:
            if isinstance(entry, str):
                name = entry
            elif isinstance(entry, dict):
                name = entry.get("name") or entry.get("material") or entry.get("id")
            else:
                name = None
            if isinstance(name, str):
                requested.append(name)
        # The planner names materials by slot (e.g. "P.M", "P.E[1].E.text"), but
        # inputs are keyed by base component ("M", "E.text"). Map slot-qualified
        # requests to their base component so the requested material is actually
        # attached instead of silently dropped.
        requested_bases = {_base_material_name(name) for name in requested}
        materials: dict[str, Any] = {}

        def attach(base: str) -> None:
            if base in inputs:
                materials.setdefault(base, inputs[base])

        for base in requested_bases:
            attach(base)
        if agent == "E.text-Finder":
            attach("D.text")
        if agent in {"M-Inducer", "T-Inducer"}:
            attach("paired_examples")
        # The E.text-Finder always benefits from the pattern components.
        if agent == "E.text-Finder":
            for key in ("M", "T", "E.tab", "E.text"):
                attach(key)
        return materials

    def _specialist_user_prompt(
        self,
        task_packet: dict[str, Any],
        materials: dict[str, Any],
        blackboard: BlackboardV2,
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "task_packet": task_packet,
            "materials": materials,
            "blackboard": blackboard.to_dict(),
        }
        if extra:
            payload["extra"] = extra
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _invoke_m_or_t(
        self,
        agent: str,
        task_packet: dict[str, Any],
        inputs: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        materials = self._resolve_requested_materials(task_packet, inputs, agent)
        parsed, raw, usage = self._call_json(
            SPECIALIST_SYSTEMS[agent],
            self._specialist_user_prompt(task_packet, materials, blackboard),
        )
        if not isinstance(parsed, dict):
            raise ValueError(f"{agent} did not return a JSON object")
        outputs = parsed.get("outputs", {}) if isinstance(parsed.get("outputs"), dict) else {}
        for key in ("M_hat", "T_hat"):
            if isinstance(outputs.get(key), str):
                blackboard.working_memory[key] = outputs[key]
        for key in ("M", "T"):
            if isinstance(outputs.get(key), str):
                slot = self._slot_for_component(key, task_packet)
                blackboard.working_memory["candidate_assignments"][slot] = outputs[key]
        self._trace(agent, sorted(materials), {"parsed": parsed, "raw_output": raw, "usage": usage})
        return parsed

    def _slot_for_component(self, component: str, task_packet: dict[str, Any]) -> str:
        query_context = task_packet.get("query_context", {})
        slots = query_context.get("variable_slots", [])
        if isinstance(slots, list):
            for slot in slots:
                if str(slot).endswith(f".{component}"):
                    return str(slot)
        return f"P.{component}"

    def _extract_from_chunk(
        self,
        chunk: ReportChunk,
        task_packet: dict[str, Any],
        materials: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        try:
            chunk_materials = dict(materials)
            chunk_materials["D.text"] = chunk.text
            parsed, raw, usage = self._call_json(
                SPECIALIST_SYSTEMS["E.text-Finder"],
                self._specialist_user_prompt(
                    task_packet,
                    chunk_materials,
                    blackboard,
                    {"chunk_id": chunk.chunk_id, "instruction": "Inspect this chunk only."},
                ),
            )
            candidates_by_slot = parsed.get("candidates", {}) if isinstance(parsed, dict) else {}
            valid: dict[str, list[dict[str, Any]]] = {}
            if isinstance(candidates_by_slot, dict):
                for slot, candidates in candidates_by_slot.items():
                    if not isinstance(candidates, list):
                        continue
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        # No verbatim gate here: the E.text-Finder verifies its own
                        # spans (its prompt requires an exact substring) and the
                        # judge enforces verbatim at evaluation. We keep the span as
                        # the model returned it.
                        excerpt = str(candidate.get("excerpt", "")).strip()
                        if not excerpt:
                            continue
                        valid.setdefault(str(slot), []).append(
                            {
                                "excerpt": excerpt,
                                "chunk_id": chunk.chunk_id,
                                "confidence": float(candidate.get("confidence", 0.0) or 0.0),
                                "match_basis": candidate.get("match_basis", []),
                                "notes": candidate.get("notes", ""),
                            }
                        )
            return {
                "chunk_id": chunk.chunk_id,
                "candidates": valid,
                "raw_output": raw,
                "usage": usage,
                "error": None,
            }
        except Exception as error:
            return {
                "chunk_id": chunk.chunk_id,
                "candidates": {},
                "raw_output": "",
                "usage": {},
                "error": str(error),
            }

    def _invoke_e_text_finder(
        self,
        task_packet: dict[str, Any],
        inputs: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        materials = self._resolve_requested_materials(task_packet, inputs, "E.text-Finder")
        report_text = materials.get("D.text")
        if not isinstance(report_text, str) or not report_text:
            raise ValueError("E.text-Finder requires D.text")
        chunks = chunk_report(report_text, self.config.chunk_chars, self.config.chunk_overlap_chars)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self._extract_from_chunk, chunk, task_packet, materials, blackboard)
                for chunk in chunks
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["chunk_id"])
        # Map output -> one global pool of distinct verbatim candidates (across all
        # chunks and slots), so the reduce step can see every option at once.
        pool: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for result in results:
            for _slot, candidates in result["candidates"].items():
                for candidate in candidates:
                    norm = " ".join(candidate["excerpt"].casefold().split())
                    if norm in seen:
                        index = seen[norm]
                        if candidate["confidence"] > pool[index]["confidence"]:
                            pool[index] = {
                                "excerpt": candidate["excerpt"],
                                "confidence": candidate["confidence"],
                                "chunk_id": candidate["chunk_id"],
                            }
                        continue
                    seen[norm] = len(pool)
                    pool.append(
                        {
                            "excerpt": candidate["excerpt"],
                            "confidence": candidate["confidence"],
                            "chunk_id": candidate["chunk_id"],
                        }
                    )
        pool.sort(key=lambda candidate: candidate["confidence"], reverse=True)

        target_slots = list(self.etext_slots)
        k = len(target_slots) or 1
        reducer_selection = self._reduce_pool(pool, materials, blackboard, k)
        picks = list(reducer_selection)
        # Backfill from the confidence-ordered pool if the reducer under-selected,
        # so each requested slot still receives a verbatim excerpt.
        for candidate in pool:
            if len(picks) >= k:
                break
            if candidate["excerpt"] not in picks:
                picks.append(candidate["excerpt"])

        if target_slots:
            for slot, excerpt in zip(target_slots, picks):
                blackboard.working_memory["candidate_assignments"][slot] = excerpt
        chunk_for = {candidate["excerpt"]: candidate["chunk_id"] for candidate in pool}
        self.selected_chunks = sorted({chunk_for[ex] for ex in picks[:k] if ex in chunk_for})

        parsed = {
            "agent": "E.text-Finder",
            "status": "success" if picks else "no_match",
            "pool_size": len(pool),
            "selected": picks[:k],
        }
        self._trace(
            "E.text-Finder",
            sorted(materials),
            {"parsed": parsed, "pool": pool, "reducer_selection": reducer_selection, "chunk_results": results},
        )
        return parsed

    def _reduce_pool(
        self,
        pool: list[dict[str, Any]],
        materials: dict[str, Any],
        blackboard: BlackboardV2,
        k: int,
    ) -> list[str]:
        """Pick the top-k verbatim excerpts from the pool by pool index.

        Selection is by index so the chosen text is always the exact pool excerpt;
        the model cannot paraphrase a span into the final answer.
        """
        if not pool:
            return []
        m_text = materials.get("M") or blackboard.working_memory.get("M_hat")
        t_text = materials.get("T") or blackboard.working_memory.get("T_hat")
        e_tab = materials.get("E.tab")
        guidance_lines = []
        if isinstance(m_text, str) and m_text.strip():
            guidance_lines.append(f"M (mathematical definition):\n{m_text}")
        if isinstance(t_text, str) and t_text.strip():
            guidance_lines.append(f"T (natural-language description):\n{t_text}")
        if isinstance(e_tab, str) and e_tab.strip():
            guidance_lines.append(f"E.tab (compact table):\n{e_tab}")
        guidance = "\n\n".join(guidance_lines) or "(no explicit guidance provided)"
        numbered = "\n".join(f"{index + 1}. {item['excerpt']}" for index, item in enumerate(pool))
        user = (
            f"GUIDANCE:\n{guidance}\n\n"
            f"Select the top K = {k} excerpts.\n\n"
            f"POOL:\n{numbered}"
        )
        try:
            parsed, raw, usage = self._call_json(REDUCER_SYSTEM, user)
        except Exception as error:  # fall back to confidence order on any failure
            self._trace("E.text-Reducer", ["pool"], {"error": str(error), "pool_size": len(pool)})
            return []
        selected = parsed.get("selected") if isinstance(parsed, dict) else None
        picks: list[str] = []
        if isinstance(selected, list):
            for item in selected:
                pid = item.get("pool_id") if isinstance(item, dict) else item
                if isinstance(pid, bool) or not isinstance(pid, (int, float)):
                    continue
                index = int(pid) - 1
                if 0 <= index < len(pool):
                    excerpt = pool[index]["excerpt"]
                    if excerpt not in picks:
                        picks.append(excerpt)
                if len(picks) >= k:
                    break
        self._trace(
            "E.text-Reducer",
            ["pool", "M", "T", "E.tab"],
            {"parsed": parsed, "raw_output": raw, "usage": usage, "pool_size": len(pool), "selected": picks},
        )
        return picks

    def _invoke_specialist(
        self,
        action: dict[str, Any],
        inputs: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        agent = action.get("agent")
        task_packet = action.get("task_packet")
        if not isinstance(agent, str) or agent not in SPECIALIST_SYSTEMS:
            raise ValueError(f"Unknown specialist agent: {agent}")
        if not isinstance(task_packet, dict):
            raise ValueError("Framework Agent invoke_agent action missing task_packet")
        if agent in {"M-Inducer", "T-Inducer"}:
            return self._invoke_m_or_t(agent, task_packet, inputs, blackboard)
        if agent == "E.text-Finder":
            return self._invoke_e_text_finder(task_packet, inputs, blackboard)
        raise ValueError(f"Unhandled specialist agent: {agent}")

    def _normalize_final(
        self,
        task: dict[str, Any],
        parsed_query: ParsedQueryV2,
        final_assignments: dict[str, Any],
    ) -> dict[str, Any]:
        experiment = str(task["experiment"])
        if experiment == "1":
            values = [
                str(final_assignments.get(slot, ""))
                for slot in parsed_query.variable_slots
                if slot.endswith(".E.text")
            ]
            if not values:
                values = [str(value) for key, value in final_assignments.items() if key.endswith(".E.text")]
            while len(values) < int(task.get("target_count", 3)):
                values.append("")
            return {"examples": [{"excerpt": value} for value in values[: int(task.get("target_count", 3))]]}
        if experiment == "2":
            value = ""
            for slot in parsed_query.variable_slots:
                if slot.endswith(".E.text"):
                    value = str(final_assignments.get(slot, ""))
                    break
            if not value:
                value = str(next(iter(final_assignments.values()), ""))
            return {"excerpt": value}
        if experiment == "3":
            m_value = ""
            t_value = ""
            for slot, value in final_assignments.items():
                if slot.endswith(".M"):
                    m_value = str(value)
                if slot.endswith(".T"):
                    t_value = str(value)
            return {"M": m_value, "T": t_value}
        raise ValueError(f"Unknown experiment: {experiment}")

    def run(self, task: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        self.trace = []
        self.selected_chunks = []
        parsed_query = self.parser.parse(task["framework_query"])
        # E.text target slots, in order, drive how many excerpts the reducer picks.
        self.etext_slots = [s for s in parsed_query.variable_slots if s.endswith(".E.text")]
        blackboard = BlackboardV2()
        self._trace(
            "FrameworkAgent.initialize",
            ["framework_query"],
            {
                "parsed_query": parsed_query.to_dict(),
                "material_inventory": self._material_inventory(inputs),
            },
        )
        last_action: dict[str, Any] | None = None
        prev_finder_snapshot: str | None = None
        consecutive_finder = 0
        for _ in range(self.config.max_steps):
            action = self._call_framework_agent(task, parsed_query, inputs, blackboard)
            last_action = action
            if action.get("action") == "return_final":
                assignments = action.get("final_assignments", {})
                if not isinstance(assignments, dict):
                    raise ValueError("return_final action missing final_assignments object")
                # Use the specialist outputs directly: take values from the
                # blackboard's candidate assignments and fall back to the
                # planner's value only for slots the specialists did not fill.
                # This keeps verbatim E.text spans intact instead of letting the
                # planner retype (and possibly corrupt) them.
                final = dict(blackboard.working_memory.get("candidate_assignments", {}))
                for slot, value in assignments.items():
                    final.setdefault(slot, value)
                parsed_output = self._normalize_final(task, parsed_query, final)
                return {
                    "parsed_output": parsed_output,
                    "raw_output": json.dumps(parsed_output, ensure_ascii=False),
                    "agent_trace": self.trace,
                    "selected_chunks": self.selected_chunks,
                    "framework_agent_final": action,
                }
            if action.get("action") != "invoke_agent":
                raise ValueError(f"Unknown Framework Agent action: {action.get('action')}")
            result = self._invoke_specialist(action, inputs, blackboard)
            blackboard.trace.append(
                {
                    "agent": action.get("agent"),
                    "reason": action.get("reason"),
                    "result_summary": _summarize_result(result),
                }
            )
            # Safety bound: stop re-running the finder when it makes no progress
            # (identical candidate set on a repeat call, or two finder calls in a
            # row). The planner should return final once candidates exist.
            if action.get("agent") == "E.text-Finder":
                consecutive_finder += 1
                snapshot = json.dumps(
                    blackboard.working_memory.get("candidate_assignments", {}),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                if snapshot == prev_finder_snapshot or consecutive_finder >= 2:
                    break
                prev_finder_snapshot = snapshot
            else:
                consecutive_finder = 0
        # Give up gracefully instead of discarding work: return the candidates
        # produced so far so a hard task still yields its best attempt.
        final = dict(blackboard.working_memory.get("candidate_assignments", {}))
        parsed_output = self._normalize_final(task, parsed_query, final)
        unresolved = [s for s in parsed_query.variable_slots if s not in final]
        return {
            "parsed_output": parsed_output,
            "raw_output": json.dumps(parsed_output, ensure_ascii=False),
            "agent_trace": self.trace,
            "selected_chunks": self.selected_chunks,
            "framework_agent_final": {
                "action": "max_steps_best_effort",
                "last_action": last_action,
                "unresolved_slots": unresolved,
            },
        }


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": result.get("status"),
    }
    if isinstance(result.get("outputs"), dict):
        summary["outputs"] = {
            key: "available" if value else "empty"
            for key, value in result["outputs"].items()
        }
    if isinstance(result.get("candidates"), dict):
        summary["candidate_slots"] = {
            key: len(value) if isinstance(value, list) else 0
            for key, value in result["candidates"].items()
        }
    if isinstance(result.get("slot_verdicts"), dict):
        summary["slot_verdicts"] = {
            key: value.get("verdict") if isinstance(value, dict) else value
            for key, value in result["slot_verdicts"].items()
        }
    return summary
