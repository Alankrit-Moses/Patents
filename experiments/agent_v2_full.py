from __future__ import annotations

from typing import Any

from .agent_v2 import BlackboardV2, FrameworkAgentV2


FULL_FINDER_SYSTEM = """You are the E.text-Finder agent. You read the ENTIRE report (D.text) in one pass and return the final selection yourself. There is no map step, no reduce step, and no later selector: your output is the final answer for the requested E.text variables.

Your goal is to find the K spans in D.text that best match the target. K is given in the task. The target is defined by the guidance you are given: M (the mathematical definition), T (the description), and/or M_hat / T_hat (internal hypotheses), or, when those are absent, an E.tab whose entities, metric, years/interval, values, or trend the span should describe. Matching the target is the primary objective; use the guidance to judge which spans fit best.

Do not coordinate other agents. Do not induce final M or T. Do not produce E.tab. Do not solve unrelated query variables.

Framework context:
- E is a set of example pairs e_i = (E.text_i, E.tab_i).
- E.text_i is a textual example found in D.text.
- When the query contains (?, E1.tab), return the E.text span that corresponds to that specific E.tab instance.

How to rank by fit:
- A span fits when it reflects the pattern behavior in the guidance. For a trajectory/trend pattern this includes a described increase, decrease, acceleration, slowdown, reversal, or other change in a quantity over time. You do NOT need an explicit "before vs after" phrase in the text; a clearly described trend counts as an instance.
- Rank a clear change or trend above a weak, vague, or no-change statement. Include a "stable" / no-change span only if there are not at least K better spans.
- For a compact table E.tab, rank highest the span that describes the same entities, metric, interval, values, or trend.
- Prefer strong, distinct instances. Do not select near-duplicates that say essentially the same thing; prefer a different span over a duplicate.

Selection rule: return exactly K spans in best-first order whenever the report contains at least K valid distinct spans. Return fewer than K only when the report itself does not contain K valid distinct spans. Never return an empty list.

Verbatim - you verify this yourself, there is no external checker:
- Each returned span must be an exact contiguous substring of D.text: copied character-for-character, including punctuation and line breaks. Do not paraphrase, summarize, stitch together non-contiguous text, reorder, or fix typos.
- Before returning, re-read the report and confirm each span appears in it verbatim. If a span has drifted from the source, correct it to match the report exactly. Do not return a span you cannot locate verbatim in D.text.

Other constraints:
- Do not return table text as E.text.
- Do not return a bare heading, an isolated number, or a citation as the entire span.

Return JSON only:
{
  "agent": "E.text-Finder",
  "status": "success | partial",
  "selected": [
    {
      "excerpt": "exact verbatim substring of D.text",
      "confidence": 0.0,
      "match_basis": ["entity", "metric", "years", "trend", "pattern_behavior"],
      "notes": "brief note"
    }
  ]
}"""


class FrameworkAgentV2FullContext(FrameworkAgentV2):
    """Setup C variant with a single-pass, full-context E.text-Finder.

    The planner (Framework Agent), M-Inducer, T-Inducer, blackboard, and control
    loop are inherited unchanged. Only the E.text-Finder differs: instead of the
    map-reduce search over overlapping report chunks, the finder receives the
    full D.text in ONE call and returns its own final ranked selection. This
    isolates the retrieval strategy (map-reduce vs full-context) as the only
    variable between the two Setup C variants.

    Like Setups A and B, the full report is never silently truncated: if the
    report plus prompt exceeds the configured context limit, the call raises so
    the run records an error instead of distorting the comparison.
    """

    def _check_context_fits(self, prompt_chars: int) -> None:
        estimated = int(prompt_chars / self.config.chars_per_token_estimate) + 32
        reserve = self.config.generator.max_output_tokens
        if estimated + reserve > self.config.context_limit_tokens:
            raise ValueError(
                f"Estimated full-context finder prompt ({estimated} tokens) plus output "
                f"reserve ({reserve}) exceeds context_limit_tokens="
                f"{self.config.context_limit_tokens}; inputs were not truncated"
            )

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

        target_slots = list(self.etext_slots)
        k = len(target_slots) or 1
        user = self._specialist_user_prompt(
            task_packet,
            materials,
            blackboard,
            {
                "mode": "full-context",
                "requested_count": k,
                "instruction": (
                    f"Read the full report in materials['D.text'] and return the {k} "
                    "best-fitting verbatim span(s) in best-first order."
                ),
            },
        )
        self._check_context_fits(len(FULL_FINDER_SYSTEM) + len(user))
        parsed, raw, usage = self._call_json(FULL_FINDER_SYSTEM, user)
        if not isinstance(parsed, dict):
            raise ValueError("E.text-Finder did not return a JSON object")

        picks: list[str] = []
        seen: set[str] = set()

        def add(excerpt: Any) -> None:
            text = str(excerpt or "").strip()
            norm = " ".join(text.casefold().split())
            if text and norm not in seen:
                seen.add(norm)
                picks.append(text)

        selected = parsed.get("selected")
        if isinstance(selected, list):
            for item in selected:
                if isinstance(item, dict):
                    add(item.get("excerpt"))
        # Tolerate the chunk-finder "candidates" shape if the model emits it.
        candidates_by_slot = parsed.get("candidates")
        if not picks and isinstance(candidates_by_slot, dict):
            for candidates in candidates_by_slot.values():
                if isinstance(candidates, list):
                    for candidate in candidates:
                        if isinstance(candidate, dict):
                            add(candidate.get("excerpt"))
        picks = picks[:k]

        if target_slots:
            for slot, excerpt in zip(target_slots, picks):
                blackboard.working_memory["candidate_assignments"][slot] = excerpt
        self.selected_chunks = ["full-report"]

        result = {
            "agent": "E.text-Finder",
            "status": "success" if picks else "no_match",
            "mode": "full-context",
            "selected": picks,
        }
        self._trace(
            "E.text-Finder",
            sorted(materials),
            {
                "parsed": result,
                "raw_output": raw,
                "usage": usage,
                "mode": "full-context",
            },
        )
        return result
