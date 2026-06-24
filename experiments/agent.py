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


def _split_top_level(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "({[":
            depth += 1
        elif char in ")} ]".replace(" ", ""):
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


class FrameworkQueryParser:
    """Parses the small typed query language used by the prototype."""

    def parse(self, query: str) -> dict[str, Any]:
        compact = " ".join(query.split())
        data_match = re.search(r"D:\(([^()]*)\)", compact)
        pattern_match = re.search(r"P:\((.*)\)\s*$", compact)
        if not data_match or not pattern_match:
            raise ValueError(f"Invalid framework query: {query}")
        data_slots = _split_top_level(data_match.group(1))
        pattern_slots = _split_top_level(pattern_match.group(1))
        if len(data_slots) != 2 or len(pattern_slots) != 3:
            raise ValueError(f"Unexpected framework arity: {query}")

        result: dict[str, Any] = {
            "slots": {
                "D.text": _slot_state(data_slots[0]),
                "D.tab": _slot_state(data_slots[1]),
                "M": _slot_state(pattern_slots[0]),
                "T": _slot_state(pattern_slots[1]),
            },
            "requested_count": 1,
        }
        example_spec = pattern_slots[2]
        count_match = re.search(r"\[(\d+)\]", example_spec)
        if count_match:
            result["requested_count"] = int(count_match.group(1))
        pairs = re.findall(r"\(([^(),{}]+),([^(),{}]+)\)", example_spec)
        text_states = [_slot_state(left.strip()) for left, _ in pairs]
        tab_states = [_slot_state(right.strip()) for _, right in pairs]

        def combine(states: list[str]) -> str:
            if "target" in states:
                return "target"
            if "known" in states:
                return "known"
            return "irrelevant"

        result["slots"]["E.text"] = combine(text_states)
        result["slots"]["E.tab"] = combine(tab_states)
        result["known"] = [key for key, state in result["slots"].items() if state == "known"]
        result["targets"] = [key for key, state in result["slots"].items() if state == "target"]
        result["irrelevant"] = [
            key for key, state in result["slots"].items() if state == "irrelevant"
        ]
        result["available_data"] = [
            key for key in ("D.text", "D.tab") if result["slots"][key] == "known"
        ]
        return result


@dataclass
class Blackboard:
    known: dict[str, Any]
    targets: list[str]
    irrelevant: list[str]
    available_data: list[str]
    internal: dict[str, Any] = field(
        default_factory=lambda: {
            "M_hat": None,
            "T_hat": None,
            "TabSignal": None,
            "Candidate_E_text": None,
            "verification_scores": None,
        }
    )
    candidates: list[dict[str, Any]] = field(default_factory=list)
    verification_scores: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "known_components": sorted(self.known),
            "targets": self.targets,
            "irrelevant": self.irrelevant,
            "available_data": self.available_data,
            "internal_helper_components": {
                key: "available" if value is not None else "unset"
                for key, value in self.internal.items()
            },
        }


class FrameworkAgent:
    def __init__(self, config: HarnessConfig, client: OpenAICompatibleClient):
        self.config = config
        self.client = client
        self.parser = FrameworkQueryParser()
        self.trace: list[dict[str, Any]] = []
        self.selected_chunks: list[str] = []

    def _trace(self, action: str, input_refs: list[str], output: Any) -> None:
        if len(self.trace) >= self.config.max_steps:
            raise RuntimeError(
                f"Setup C exceeded max_steps={self.config.max_steps} before {action}"
            )
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

    def _tab_signal(self, csv_text: str) -> dict[str, Any]:
        parsed, raw, usage = self._call_json(
            "You are TabSignalAgent. Summarize only the supplied compact CSV. Do not invent rows, source data, or verified tabular evidence. Return JSON only.",
            f"""Extract the evidence signal from this compact E.tab CSV.

CSV:
{csv_text}

Return:
{{"entities":[],"metric":"","years_or_interval":"","trend":"","turning_point":null,"comparison":null,"likely_pattern_behavior":""}}""",
        )
        output = {"parsed": parsed, "raw_output": raw, "usage": usage}
        self._trace("TabSignalAgent", ["E.tab"], output)
        return parsed

    def _description_induction(self, evidence: dict[str, Any]) -> str:
        parsed, raw, usage = self._call_json(
            "You are DescriptionInductionAgent. Translate supplied formal or observed evidence into a concise report-searchable pattern description. Return JSON only.",
            f"""Produce an internal helper T_hat. Preserve all constraints and do not add facts.

Evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return {{"T_hat":"..."}}.""",
        )
        value = str(parsed.get("T_hat", "")) if isinstance(parsed, dict) else ""
        self._trace(
            "DescriptionInductionAgent",
            list(evidence),
            {"T_hat": value, "raw_output": raw, "usage": usage},
        )
        return value

    def _extract_from_chunk(
        self, chunk: ReportChunk, evidence_brief: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            parsed, raw, usage = self._call_json(
                "You are TextEvidenceAgent. Inspect one report chunk. Return only exact verbatim spans found in the chunk. A summary or rewritten quote is invalid. Return JSON only.",
                f"""Evidence brief:
{json.dumps(evidence_brief, ensure_ascii=False, indent=2)}

Report chunk {chunk.chunk_id}:
{chunk.text}

Return up to two candidates:
{{"candidates":[{{"excerpt":"exact text","local_score":1,"matched_constraints":[]}}]}}
Use an empty candidates list when no span satisfies the brief.""",
            )
            candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            valid: list[dict[str, Any]] = []
            for candidate in candidates[:2]:
                if not isinstance(candidate, dict):
                    continue
                recovered = recover_exact_span(str(candidate.get("excerpt", "")), chunk.text)
                if recovered:
                    valid.append(
                        {
                            "excerpt": recovered,
                            "chunk_id": chunk.chunk_id,
                            "local_score": int(candidate.get("local_score", 1)),
                            "matched_constraints": candidate.get("matched_constraints", []),
                        }
                    )
            return {
                "chunk_id": chunk.chunk_id,
                "candidates": valid,
                "raw_output": raw,
                "usage": usage,
                "error": None,
            }
        except Exception as error:  # preserve other chunk results
            return {
                "chunk_id": chunk.chunk_id,
                "candidates": [],
                "raw_output": "",
                "usage": {},
                "error": str(error),
            }

    def _text_evidence(
        self, report_text: str, evidence_brief: dict[str, Any]
    ) -> list[dict[str, Any]]:
        chunks = chunk_report(
            report_text, self.config.chunk_chars, self.config.chunk_overlap_chars
        )
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._extract_from_chunk, chunk, evidence_brief): chunk.chunk_id
                for chunk in chunks
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["chunk_id"])
        deduplicated: dict[str, dict[str, Any]] = {}
        for result in results:
            for candidate in result["candidates"]:
                key = " ".join(candidate["excerpt"].casefold().split())
                previous = deduplicated.get(key)
                if previous is None or candidate["local_score"] > previous["local_score"]:
                    deduplicated[key] = candidate
        candidates = sorted(
            deduplicated.values(), key=lambda item: item["local_score"], reverse=True
        )[:50]
        self._trace(
            "TextEvidenceAgent",
            ["D.text", *[key for key in evidence_brief if key != "target"]],
            {
                "chunk_count": len(chunks),
                "chunk_results": results,
                "deduplicated_candidates": candidates,
            },
        )
        return candidates

    def _verify_text_pattern(
        self, candidate: dict[str, Any], criterion: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            parsed, raw, usage = self._call_json(
                "You are TextPatternVerifier, separate from the generator. Score whether the exact excerpt satisfies the supplied pattern constraints. Return JSON only.",
                f"""Pattern constraints:
{json.dumps(criterion, ensure_ascii=False, indent=2)}

Candidate E.text:
{candidate['excerpt']}

Return {{"score":1,"matched_conditions":[],"failed_conditions":[],"rationale":""}}. Score from 1 to 5.""",
            )
            verdict = parsed if isinstance(parsed, dict) else {}
            verdict["score"] = max(1, min(5, int(verdict.get("score", 1))))
            verdict.update({"candidate": candidate, "raw_output": raw, "usage": usage})
            return verdict
        except Exception as error:
            return {
                "score": 1,
                "matched_conditions": [],
                "failed_conditions": ["verifier_error"],
                "rationale": str(error),
                "candidate": candidate,
            }

    def _verify_tab_text(
        self, candidate: dict[str, Any], csv_text: str, tab_signal: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            parsed, raw, usage = self._call_json(
                "You are TabTextVerifier, separate from the generator. Score whether the candidate excerpt describes the same observed compact table evidence. Return JSON only.",
                f"""Provided E.tab CSV:
{csv_text}

Internal TabSignal:
{json.dumps(tab_signal, ensure_ascii=False, indent=2)}

Candidate E.text:
{candidate['excerpt']}

Check entity, metric, year/interval, trend, and pattern behavior. Return {{"score":1,"matched_conditions":[],"failed_conditions":[],"rationale":""}} using score 1-5.""",
            )
            verdict = parsed if isinstance(parsed, dict) else {}
            verdict["score"] = max(1, min(5, int(verdict.get("score", 1))))
            verdict.update({"candidate": candidate, "raw_output": raw, "usage": usage})
            return verdict
        except Exception as error:
            return {
                "score": 1,
                "matched_conditions": [],
                "failed_conditions": ["verifier_error"],
                "rationale": str(error),
                "candidate": candidate,
            }

    def _verify_candidates(
        self,
        candidates: list[dict[str, Any]],
        *,
        criterion: dict[str, Any] | None = None,
        csv_text: str | None = None,
        tab_signal: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        action = "TextPatternVerifier" if criterion is not None else "TabTextVerifier"
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            if criterion is not None:
                futures = [
                    executor.submit(self._verify_text_pattern, candidate, criterion)
                    for candidate in candidates
                ]
                refs = ["Candidate_E_text", *criterion.keys()]
            else:
                assert csv_text is not None and tab_signal is not None
                futures = [
                    executor.submit(self._verify_tab_text, candidate, csv_text, tab_signal)
                    for candidate in candidates
                ]
                refs = ["Candidate_E_text", "E.tab", "TabSignal"]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(
            key=lambda item: (
                item["score"], item["candidate"].get("local_score", 1)
            ),
            reverse=True,
        )
        self._trace(action, refs, {"verdicts": results})
        return results

    def _reduce(self, verdicts: list[dict[str, Any]], count: int) -> dict[str, Any]:
        selected = verdicts[:count]
        rejected = verdicts[count:]
        output = {
            "selected": [item["candidate"] for item in selected],
            "rejected": [
                {
                    "candidate": item["candidate"],
                    "score": item["score"],
                    "reason": item.get("rationale", "lower rank"),
                }
                for item in rejected
            ],
            "reason": "Ranked by verifier score, then chunk-local score.",
        }
        self.selected_chunks = sorted(
            {candidate["chunk_id"] for candidate in output["selected"]}
        )
        self._trace("ReducerRanker", ["Candidate_E_text", "verification_scores"], output)
        return output

    def _pattern_synthesis(self, pairs: list[dict[str, str]]) -> dict[str, str]:
        parsed, raw, usage = self._call_json(
            "You are PatternSynthesisAgent. Infer one recurring structural behavior from paired textual and compact tabular examples. Return a precise operational mathematical definition and a concise description as JSON only.",
            f"""Paired examples:
{json.dumps(pairs, ensure_ascii=False, indent=2)}

Produce a definition that is neither broader nor narrower than the shared behavior. State comparability, temporal, aggregation, and boundary conditions when relevant. Return {{"M":"...","T":"..."}}.""",
        )
        output = {
            "M": str(parsed.get("M", "")) if isinstance(parsed, dict) else "",
            "T": str(parsed.get("T", "")) if isinstance(parsed, dict) else "",
        }
        self._trace(
            "PatternSynthesisAgent",
            ["E.text", "E.tab"],
            {**output, "raw_output": raw, "usage": usage},
        )
        return output

    def _definition_verifier(
        self,
        definition: dict[str, str],
        evidence_pairs: list[dict[str, str]],
    ) -> dict[str, Any]:
        parsed, raw, usage = self._call_json(
            "You are DefinitionVerifier, independent of the pattern generator. Evaluate fidelity and operationalizability. Return JSON only.",
            f"""Generated definition:
{json.dumps(definition, ensure_ascii=False, indent=2)}

Provided paired examples:
{json.dumps(evidence_pairs, ensure_ascii=False, indent=2)}

Return {{"score":1,"M_score":1,"T_score":1,"operationalizability":1,"matched_conditions":[],"failed_conditions":[],"rationale":""}}. Scores are 1-5. Judge only against the supplied pairs and do not reward polished wording that lacks a shared operational rule.""",
        )
        verdict = parsed if isinstance(parsed, dict) else {}
        for key in ("score", "M_score", "T_score", "operationalizability"):
            verdict[key] = max(1, min(5, int(verdict.get(key, 1))))
        verdict.update({"raw_output": raw, "usage": usage})
        self._trace(
            "DefinitionVerifier",
            ["M", "T", "paired_examples"],
            verdict,
        )
        return verdict

    def _repair(self, definition: dict[str, str], feedback: dict[str, Any]) -> dict[str, str]:
        safe_feedback = {
            key: feedback.get(key)
            for key in ("failed_conditions", "rationale", "M_score", "T_score", "operationalizability")
        }
        parsed, raw, usage = self._call_json(
            "You are RepairAgent. Revise the proposed definition once using only verifier feedback. Return JSON only.",
            f"""Current output:
{json.dumps(definition, ensure_ascii=False, indent=2)}

Verifier feedback:
{json.dumps(safe_feedback, ensure_ascii=False, indent=2)}

Return {{"M":"...","T":"..."}}.""",
        )
        repaired = {
            "M": str(parsed.get("M", definition["M"])) if isinstance(parsed, dict) else definition["M"],
            "T": str(parsed.get("T", definition["T"])) if isinstance(parsed, dict) else definition["T"],
        }
        self._trace(
            "RepairAgent",
            ["M", "T", "verification_scores"],
            {**repaired, "raw_output": raw, "usage": usage},
        )
        return repaired

    def run(self, task: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        self.trace = []
        self.selected_chunks = []
        parsed_query = self.parser.parse(task["framework_query"])
        known = {key: value for key, value in inputs.items() if key != "D.text"}
        blackboard = Blackboard(
            known=known,
            targets=parsed_query["targets"],
            irrelevant=parsed_query["irrelevant"],
            available_data=parsed_query["available_data"],
        )
        self._trace(
            "FrameworkOrchestrator.parse",
            ["framework_query"],
            {
                "parsed_query": parsed_query,
                "blackboard": blackboard.summary(),
                "compiled_plan": setup_c_plan_preview(task),
            },
        )

        if "E.tab" in blackboard.targets and "D.tab" not in blackboard.available_data:
            raise ValueError("Cannot produce newly verified E.tab without D.tab")

        if "E.text" in blackboard.targets:
            report_text = inputs.get("D.text")
            if not report_text:
                raise ValueError("E.text target requires available D.text")
            if "E.tab" in inputs:
                tab_signal = self._tab_signal(inputs["E.tab"])
                blackboard.internal["TabSignal"] = tab_signal
                t_hat = self._description_induction({"TabSignal": tab_signal})
                blackboard.internal["T_hat"] = t_hat
                brief = {
                    "known": ["E.tab", "TabSignal"],
                    "target": "E.text",
                    "TabSignal": tab_signal,
                    "T_hat": t_hat,
                    "instruction": "Find exact spans matching entity, metric, years, trend, and behavior.",
                }
                candidates = self._text_evidence(report_text, brief)
                blackboard.candidates = candidates
                blackboard.internal["Candidate_E_text"] = candidates
                verdicts = self._verify_candidates(
                    candidates,
                    csv_text=inputs["E.tab"],
                    tab_signal=tab_signal,
                )
            else:
                criterion = {key: inputs[key] for key in ("M", "T") if key in inputs}
                if "M" in criterion and "T" not in criterion:
                    t_hat = self._description_induction({"M": criterion["M"]})
                    blackboard.internal["T_hat"] = t_hat
                    criterion["T_hat"] = t_hat
                brief = {
                    "known": list(criterion),
                    "target": "E.text",
                    **criterion,
                    "instruction": "Find exact spans satisfying the structural pattern, not merely topical matches.",
                }
                candidates = self._text_evidence(report_text, brief)
                blackboard.candidates = candidates
                blackboard.internal["Candidate_E_text"] = candidates
                verdicts = self._verify_candidates(candidates, criterion=criterion)
            blackboard.verification_scores = verdicts
            blackboard.internal["verification_scores"] = verdicts
            reduced = self._reduce(verdicts, parsed_query["requested_count"])
            excerpts = [candidate["excerpt"] for candidate in reduced["selected"]]
            parsed_output: dict[str, Any]
            if parsed_query["requested_count"] == 1:
                parsed_output = {"excerpt": excerpts[0] if excerpts else ""}
            else:
                parsed_output = {"examples": [{"excerpt": value} for value in excerpts]}
            return {
                "parsed_output": parsed_output,
                "raw_output": json.dumps(parsed_output, ensure_ascii=False),
                "agent_trace": self.trace,
                "selected_chunks": self.selected_chunks,
            }

        if {"M", "T"} & set(blackboard.targets):
            pairs = inputs.get("paired_examples", [])
            if not pairs:
                raise ValueError("M/T synthesis requires paired examples in this prototype")
            definition = self._pattern_synthesis(pairs)
            verdict = self._definition_verifier(definition, pairs)
            if self.config.repair_once and verdict["score"] < 4:
                definition = self._repair(definition, verdict)
            requested = {
                key: definition[key] for key in ("M", "T") if key in blackboard.targets
            }
            return {
                "parsed_output": requested,
                "raw_output": json.dumps(requested, ensure_ascii=False),
                "agent_trace": self.trace,
                "selected_chunks": [],
            }

        raise ValueError(f"No supported target in {blackboard.targets}")


def setup_c_plan_preview(task: dict[str, Any]) -> list[str]:
    parsed = FrameworkQueryParser().parse(task["framework_query"])
    targets = set(parsed["targets"])
    known = set(parsed["known"])
    if "E.text" in targets and "E.tab" in known:
        return [
            "FrameworkOrchestrator.parse",
            "TabSignalAgent",
            "DescriptionInductionAgent",
            "TextEvidenceAgent(map-reduce)",
            "TabTextVerifier",
            "ReducerRanker",
        ]
    if "E.text" in targets:
        return [
            "FrameworkOrchestrator.parse",
            "DescriptionInductionAgent(optional T_hat)",
            "TextEvidenceAgent(map-reduce)",
            "TextPatternVerifier",
            "ReducerRanker",
        ]
    if {"M", "T"} & targets:
        return [
            "FrameworkOrchestrator.parse",
            "PatternSynthesisAgent",
            "DefinitionVerifier",
            "RepairAgent(at most once, if needed)",
        ]
    return ["FrameworkOrchestrator.parse", "unsupported target"]
