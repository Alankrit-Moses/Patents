from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .client import OpenAICompatibleClient
from .config import HarnessConfig
from .io_utils import extract_json
from .reports import ReportChunk, chunk_report, recover_exact_span


FRAMEWORK_AGENT_SYSTEM = """You are the Framework Agent for an agentic pattern-analysis system.

Your role is to interpret a framework query, maintain the framework state, decide which specialist agent to invoke next, and return final assignments only when the query variables are resolved and verified.

You are not the primary evidence-consuming worker. You should usually not inspect full reports or full tables directly. Instead, use the framework query, material inventory, blackboard state, and specialist-agent catalog to decide the next best subtask.

FRAMEWORK SPECIFICATION

The framework formulates analytical tasks by modularizing patterns and data into components. Analytical tasks are expressed as mappings from partially specified pattern components to desired components. The LLM system fills only the missing variables explicitly marked with ?.

Data representation:
- D = (D.text, D.tab)
- D.text is textual data, such as patent landscape reports, corporate reports, or technical documents.
- D.tab is tabular data, such as spreadsheets, relational tables, or a unified table.
- Either D.text or D.tab may be marked _ if it is irrelevant or unavailable for the query.

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
- _ means the component or data source is irrelevant, unavailable, or should not be used.
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
- Do not use any component or data source marked _.
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
- You may create internal working hypotheses such as M_hat and T_hat when useful.
- M_hat is an internal mathematical-definition hypothesis, not a final M unless the query asks for M with ?.
- T_hat is an internal textual-description hypothesis, not a final T unless the query asks for T with ?.
- Internal hypotheses can help specialists retrieve, induce, verify, or repair outputs.
- Final answers must assign only query variable slots.

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

4. Verifier
Purpose: check framework compliance and evidence support.
Can read: framework query, candidate assignments, D.text, E.tab, M, T, M_hat, T_hat, relevant materials.
Can write: verdicts, scores, issues.
Use before accepting final variable assignments.

YOUR CONTROL LOOP

At each turn:
1. Parse the framework query.
2. Identify all ? variable slots exactly where they appear.
3. Identify concrete components and data sources.
4. Identify _ components and data sources that must not be used.
5. Inspect the blackboard to see which variables are already resolved and verified.
6. Decide whether an internal helper hypothesis would improve the next step.
7. Invoke exactly one specialist agent, or return final assignments if all ? slots are resolved and accepted.
8. Request only the materials needed for that specialist call.
9. Preserve pair structure. If the query has (?, E1.tab), the produced E.text must correspond to E1.tab.
10. Preserve count structure. If the query has (?,?)[2], produce exactly two example pairs unless impossible.
11. Use verification before final acceptance.

PLANNING PRINCIPLES

Prefer dynamic decomposition over direct completion when intermediate components may improve reliability.
- If multiple E.tab instances are provided and multiple paired E.text slots are ?, you may first induce M_hat and/or T_hat from the E.tab set, then ask E.text-Finder to locate exact spans for each paired slot.
- If M and T are already provided and E.text is ?, direct E.text-Finder invocation may be sufficient.
- If only examples are provided and M or T is ?, use M-Inducer and/or T-Inducer.
- If candidate outputs exist but are unverified, invoke Verifier.
- If Verifier reports weak support or a format violation, update the blackboard and invoke a suitable specialist again.

Return JSON only.

Allowed action 1: invoke_agent
{
  "action": "invoke_agent",
  "agent": "M-Inducer | T-Inducer | E.text-Finder | Verifier",
  "reason": "brief explanation of why this is the next best step",
  "task_packet": {
    "goal": "specific subtask",
    "query_context": {
      "framework_query": "...",
      "active_pattern_ids": [],
      "variable_slots": [],
      "concrete_slots": [],
      "irrelevant_slots": []
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
  "reason": "all ? slots are resolved and verified"
}

Important:
- final_assignments must include only slots that were ? in the original query.
- Do not include M_hat or T_hat in final_assignments unless the original query had ? in M or T and the value was accepted as the final component.
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
    "E.text-Finder": """You are the E.text-Finder agent.

Your role is to locate E.text examples in D.text. E.text must be an exact contiguous span copied from the supplied textual report data.

Do not coordinate other agents. Do not induce final M or T. Do not produce E.tab. Do not solve unrelated query variables.

Framework context:
- E is a set of example pairs e_i = (E.text_i, E.tab_i).
- E.text_i is a textual example found in D.text.
- When the query contains (?, E1.tab), find the E.text value corresponding to that specific E.tab instance.
- When the query contains multiple pairs, each ? is a separate variable slot and must preserve its pairing.

Rules:
- Return exact contiguous spans from D.text only.
- Do not paraphrase.
- Do not compose a passage from multiple non-contiguous locations.
- Do not return table text as E.text.
- Do not return headings, isolated numbers, citations, or topical mentions unless the span itself describes the pattern behavior.
- Preserve variable-slot identity.
- Preserve pair identity by matching entity, metric, time interval, values, trend, and behavior where available.
- Use M, T, M_hat, or T_hat as retrieval guidance, but do not output them.

Return JSON only:
{
  "agent": "E.text-Finder",
  "status": "success | partial | no_match",
  "candidates": {
    "P.E[1].E.text": [
      {
        "excerpt": "exact span from D.text",
        "confidence": 0.0,
        "match_basis": ["entity", "metric", "years", "trend", "pattern_behavior"],
        "notes": "brief note"
      }
    ]
  }
}""",
    "Verifier": """You are the Verifier agent.

Your role is to verify candidate assignments or internal hypotheses against the framework query, supplied materials, and framework rules.

Do not coordinate other agents. Do not produce new final answers unless the task packet explicitly asks for a repaired candidate. Your default role is to judge, not solve.

Framework compliance checks:
- Does the candidate assign only slots that were ? in the original query?
- Does it avoid assigning slots marked _?
- Does it preserve pair structure and fixed counts?
- If candidate E.text is provided, is it an exact contiguous span from D.text?
- Are internal hypotheses like M_hat or T_hat kept separate from final query assignments?

Evidence support checks:
- For E.text, does the span describe the pattern behavior rather than merely mention the topic?
- If paired with E.tab, does it align with the same entities, metric, years or interval, values or trend, and pattern behavior where available?
- For M, is the definition supported by supplied examples or descriptions and pattern-level?
- For T, is the description supported by supplied examples or M and pattern-level?

Return JSON only:
{
  "agent": "Verifier",
  "status": "complete",
  "overall_verdict": "accept | revise | reject",
  "slot_verdicts": {
    "P.E[1].E.text": {
      "verdict": "accept | revise | reject",
      "compliance": "pass | fail",
      "support": "strong | partial | weak | none",
      "score": 0.0,
      "issues": [],
      "suggested_next_step": "accept | retry_retrieval | induce_M_hat | induce_T_hat | repair | ask_framework_agent"
    }
  },
  "global_issues": [],
  "notes": "brief explanation"
}""",
}


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
        return "irrelevant"
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
    irrelevant_slots: list[str]
    requested_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework_query": self.framework_query,
            "data_slots": self.data_slots,
            "patterns": self.patterns,
            "variable_slots": self.variable_slots,
            "concrete_slots": self.concrete_slots,
            "irrelevant_slots": self.irrelevant_slots,
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
        irrelevant_slots = [
            key for key, state in data_slots.items() if state == "irrelevant"
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
                    irrelevant_slots.append(slot)
            example_spec = pattern_values[2]
            if example_spec == "_":
                irrelevant_slots.append(f"{pattern_id}.E")
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
                                irrelevant_slots.append(slot)
                        pattern["E"].append(pair)
                        pair_index += 1
            patterns.append(pattern)
        return ParsedQueryV2(
            framework_query=query,
            data_slots=data_slots,
            patterns=patterns,
            variable_slots=variable_slots,
            concrete_slots=concrete_slots,
            irrelevant_slots=irrelevant_slots,
            requested_count=requested_count,
        )


@dataclass
class BlackboardV2:
    working_memory: dict[str, Any] = field(
        default_factory=lambda: {
            "M_hat": None,
            "T_hat": None,
            "candidate_assignments": {},
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
                {"name": "Verifier", "can_write": ["verdicts"]},
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
        requested = task_packet.get("materials_needed", [])
        if not isinstance(requested, list):
            requested = []
        materials: dict[str, Any] = {}
        for name in requested:
            if name in inputs:
                materials[name] = inputs[name]
            elif name.startswith("E.tab") and "E.tab" in inputs:
                materials[name] = inputs["E.tab"]
            elif name.startswith("E.text") and "E.text" in inputs:
                materials[name] = inputs["E.text"]
        if agent == "E.text-Finder" and "D.text" in inputs:
            materials.setdefault("D.text", inputs["D.text"])
        if agent in {"M-Inducer", "T-Inducer", "Verifier"} and "paired_examples" in inputs:
            materials.setdefault("paired_examples", inputs["paired_examples"])
        for key in ("M", "T", "E.tab", "E.text"):
            if key in inputs and (key in requested or agent in {"E.text-Finder", "Verifier"}):
                materials.setdefault(key, inputs[key])
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
                        recovered = recover_exact_span(str(candidate.get("excerpt", "")), chunk.text)
                        if recovered:
                            valid.setdefault(str(slot), []).append(
                                {
                                    "excerpt": recovered,
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
        deduped: dict[str, dict[str, Any]] = {}
        for result in results:
            for slot, candidates in result["candidates"].items():
                for candidate in candidates:
                    key = slot + "::" + " ".join(candidate["excerpt"].casefold().split())
                    previous = deduped.get(key)
                    if previous is None or candidate["confidence"] > previous["confidence"]:
                        deduped[key] = {"slot": slot, **candidate}
        by_slot: dict[str, list[dict[str, Any]]] = {}
        for item in sorted(deduped.values(), key=lambda value: value["confidence"], reverse=True):
            slot = item.pop("slot")
            by_slot.setdefault(slot, []).append(item)
        for slot, candidates in by_slot.items():
            if candidates:
                blackboard.working_memory["candidate_assignments"][slot] = candidates[0]["excerpt"]
        self.selected_chunks = sorted(
            {
                candidate["chunk_id"]
                for candidates in by_slot.values()
                for candidate in candidates[:1]
            }
        )
        parsed = {
            "agent": "E.text-Finder",
            "status": "success" if by_slot else "no_match",
            "candidates": by_slot,
        }
        self._trace(
            "E.text-Finder",
            sorted(materials),
            {"parsed": parsed, "chunk_results": results},
        )
        return parsed

    def _invoke_verifier(
        self,
        task_packet: dict[str, Any],
        inputs: dict[str, Any],
        blackboard: BlackboardV2,
    ) -> dict[str, Any]:
        materials = self._resolve_requested_materials(task_packet, inputs, "Verifier")
        candidate_assignments = blackboard.working_memory.get("candidate_assignments", {})
        materials["candidate_assignments"] = candidate_assignments
        parsed, raw, usage = self._call_json(
            SPECIALIST_SYSTEMS["Verifier"],
            self._specialist_user_prompt(task_packet, materials, blackboard),
        )
        if not isinstance(parsed, dict):
            raise ValueError("Verifier did not return a JSON object")
        blackboard.verification_results.append(parsed)
        slot_verdicts = parsed.get("slot_verdicts", {})
        if isinstance(slot_verdicts, dict):
            for slot, verdict in slot_verdicts.items():
                if (
                    isinstance(verdict, dict)
                    and verdict.get("verdict") == "accept"
                    and slot in candidate_assignments
                ):
                    blackboard.resolved_variables[slot] = candidate_assignments[slot]
        if parsed.get("overall_verdict") == "accept":
            for slot, value in candidate_assignments.items():
                blackboard.resolved_variables.setdefault(slot, value)
        self._trace("Verifier", sorted(materials), {"parsed": parsed, "raw_output": raw, "usage": usage})
        return parsed

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
        return self._invoke_verifier(task_packet, inputs, blackboard)

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
        for _ in range(self.config.max_steps):
            action = self._call_framework_agent(task, parsed_query, inputs, blackboard)
            last_action = action
            if action.get("action") == "return_final":
                assignments = action.get("final_assignments", {})
                if not isinstance(assignments, dict):
                    raise ValueError("return_final action missing final_assignments object")
                parsed_output = self._normalize_final(task, parsed_query, assignments)
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
        raise RuntimeError(f"Setup C v2 exceeded max_steps={self.config.max_steps}; last action={last_action}")


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
