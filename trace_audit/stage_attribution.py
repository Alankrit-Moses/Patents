#!/usr/bin/env python3
"""Stage attribution for the example-alignment (Q2) executions, read off the recorded traces.

Each Q2 task holds out one held-out passage, its e.text. For every execution, this script compares that passage with what
the trace records: the E.text-Finder's result for each report chunk, the candidate pool, the selector's
choice, and the excerpt that was returned. An execution falls into one of four stages:

  A  the returned excerpt comes from the held-out passage
  B  the held-out passage was in the candidate pool, but another candidate was returned
  C  the finder read the chunk that holds the held-out passage and did not propose it
  D  the finder's call on that chunk failed

Two texts are taken to be the same place in the report when they share at least 100 consecutive characters
(or 60% of a shorter held-out passage), or when a candidate of 60 characters or more lies inside the curated
passage. Another passage can also describe the same instance, so the stage says where the held-out passage
was lost, and the judges' scores say whether the output was still acceptable.

  python3 -I stage_attribution.py                                # summary for gpt-oss-high
  python3 -I stage_attribution.py --model gpt-4.1 --write        # also write stage_attribution_gpt-4.1.csv
  python3 -I stage_attribution.py --show exp2-P1-E3:C:sample-1   # print one execution step by step

Place this folder at the root of the repository, or set PATTERNS_REPO to the repository path. The second
judge's scores are read from opus_judge/results/claude-opus-5 when that folder is present.
"""
from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_repo() -> Path:
    if os.environ.get("PATTERNS_REPO"):
        return Path(os.environ["PATTERNS_REPO"]).resolve()
    for candidate in (HERE.parent, HERE.parent / "repo_latest"):
        if (candidate / "experiments" / "evaluation.py").exists():
            return candidate.resolve()
    sys.exit("cannot find the study repository; set PATTERNS_REPO to its path")


REPO = _find_repo()
sys.path.append(str(REPO))
from experiments.io_utils import read_text  # noqa: E402
from experiments.reports import chunk_report  # noqa: E402
from experiments.task_construction import extract_saved_excerpt  # noqa: E402

MODELS = ["gpt-4.1", "gpt-oss-high", "gpt-oss-medium", "gpt-5.6-luna"]
STAGES = {
    "A": "returned excerpt comes from the held-out passage",
    "B": "held-out passage in the candidate pool, another candidate returned",
    "C": "finder read the chunk holding the held-out passage, did not propose it",
    "D": "finder call on that chunk failed",
}
SECOND_JUDGE = next((p for p in (REPO / "opus_judge" / "results" / "claude-opus-5",
                                 HERE.parent / "opus_judge" / "results" / "claude-opus-5") if p.exists()), None)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text) -> str:
    return " ".join(str(text).split())


def shared(a, b) -> int:
    """Length of the longest verbatim span two texts share (whitespace and case normalised)."""
    a, b = norm(a).lower(), norm(b).lower()
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def same_place(text, curated) -> bool:
    s, n = shared(text, curated), len(norm(text))
    return s >= 100 or s >= 0.6 * len(norm(curated)) or (n >= 60 and s >= 0.9 * n)


def plan(record) -> list[str]:
    steps = []
    for step in record["agent_trace"]:
        if step["action"] == "FrameworkAgent":
            action = step["output"].get("action", {})
            steps.append(action.get("agent") if action.get("action") == "invoke_agent" else action.get("action"))
    return steps


def returned_excerpt(record) -> str:
    output = record["parsed_output"]
    if output.get("examples"):
        return output["examples"][0].get("excerpt", "")
    return output.get("excerpt", "")


def candidates_of(chunk_result) -> list:
    found = chunk_result.get("candidates")
    if isinstance(found, dict):
        return next((value for value in found.values() if isinstance(value, list)), [])
    return found or []


def curated_chunks(report: str, curated: str, n_chunks: int) -> list[str]:
    """Ids of the chunks that hold the start of the held-out passage. Chunk boundaries are recomputed with the
    released chunker; the chunk size is the one that reproduces the number of chunks the run recorded."""
    tokens = norm(curated).split()
    start = None
    for length in (len(tokens), 40, 25, 12):
        match = re.search(r"\s+".join(re.escape(token) for token in tokens[:length]), report)
        if match:
            start = match.start()
            break
    if start is None:
        return []
    for size, overlap in ((50000, 3000), (16000, 2000), (50000, 2000), (16000, 3000)):
        chunks = chunk_report(report, size, overlap)
        if len(chunks) == n_chunks:
            return [c.chunk_id for c in chunks if c.start_char <= start < c.end_char]
    return []


def records(model: str):
    tasks = {t["task_id"]: t for name in ("tasks.jsonl", "p3.jsonl") for t in load(REPO / "experiments" / "tasks" / name)}
    files = [("results/%s/outputs.jsonl" % model, REPO / "results" / model / "outputs.jsonl"),
             ("results/final/%s/p3-outputs.jsonl" % model, REPO / "results" / "final" / model / "p3-outputs.jsonl")]
    outputs = [(name, record) for name, path in files for record in load(path)]

    def scores(base):
        if base is None:
            return {}
        return {r["sample_id"]: r["metrics"] for kind in ("p1-p2", "p3") for r in load(base / model / f"{kind}-evaluations.jsonl")}

    return tasks, outputs, scores(REPO / "results" / "final"), scores(SECOND_JUDGE)


def attribute(model: str) -> list[dict]:
    tasks, outputs, first, second = records(model)
    rows = []
    for file_name, record in outputs:
        if str(record["experiment"]) != "2":
            continue
        task = tasks[record["task_id"]]
        curated = extract_saved_excerpt(REPO / task["gold_e_text_path"])
        report = read_text(REPO / task["report_path"])
        finder_calls = [step["output"] for step in record["agent_trace"] if step["action"] == "E.text-Finder"]
        last = finder_calls[-1] if finder_calls else {}
        pool, chunk_results = last.get("pool", []), last.get("chunk_results", [])
        by_chunk = {c["chunk_id"]: c for c in chunk_results}
        holding = curated_chunks(report, curated, len(chunk_results))
        positions = [i for i, candidate in enumerate(pool, 1) if same_place(candidate["excerpt"], curated)]
        returned = returned_excerpt(record)
        if same_place(returned, curated):
            stage = "A"
        elif positions:
            stage = "B"
        elif holding and all(by_chunk.get(chunk, {}).get("error") for chunk in holding):
            stage = "D"
        else:
            stage = "C"
        sid = record["sample_id"]
        rows.append({
            "sample_id": sid,
            "file": file_name,
            "pattern": record["pattern_id"],
            "stage": stage,
            "plan": " > ".join(str(s) for s in plan(record)),
            "pool_size": len(pool),
            "pool_positions_of_held_out_passage": " ".join(map(str, positions)),
            "chunks": len(chunk_results),
            "chunk_holding_held_out_passage": " ".join(holding),
            "failed_chunk_calls": " ".join(c["chunk_id"] for c in chunk_results if c.get("error")),
            "failed_chunk_call_in_any_attempt": any(c.get("error") for call in finder_calls for c in call.get("chunk_results", [])),
            "selector_call_failed": any(s["action"] == "E.text-Reducer" and s["output"].get("error") for s in record["agent_trace"]),
            "returned_chars": len(norm(returned)),
            "chars_shared_with_held_out_passage": shared(returned, curated),
            "held_out_chars": len(norm(curated)),
            "score_gpt_6_1_terra": first[sid]["E_text_alignment_score"],
            "score_claude_opus_5": second[sid]["E_text_alignment_score"] if second else "",
        })
    return rows


def band(row) -> str:
    values = [v for v in (row["score_gpt_6_1_terra"], row["score_claude_opus_5"]) if v != ""]
    return "high" if min(values) >= 4 else ("low" if max(values) <= 2 else "mixed")


def summary(rows: list[dict]) -> str:
    lines = [f"{'stage':72s} runs  all judges >= 4  all judges <= 2"]
    for stage, text in STAGES.items():
        own = [r for r in rows if r["stage"] == stage]
        lines.append(f"{stage}  {text:68s} {len(own):4d} {sum(band(r) == 'high' for r in own):17d} {sum(band(r) == 'low' for r in own):16d}")
    low = [r for r in rows if band(r) == "low"]
    lines.append(f"executions {len(rows)}; low-scoring {len(low)}, of which retrieval stage (C or D) {sum(r['stage'] in 'CD' for r in low)}, "
                 f"selection stage (B) {sum(r['stage'] == 'B' for r in low)}; with a failed chunk call in any retrieval attempt {sum(r['failed_chunk_call_in_any_attempt'] for r in rows)}")
    return "\n".join(lines)


def show(model: str, sample_id: str, width: int = 200) -> None:
    tasks, outputs, first, second = records(model)
    found = [(name, record) for name, record in outputs if record["sample_id"] == sample_id]
    if not found:
        sys.exit(f"no record {sample_id} for {model}")
    file_name, record = found[0]
    task = tasks[record["task_id"]]
    print(f"{file_name}  sample_id {sample_id}")
    print(f"scores: GPT-6.1 Terra {first[sample_id]}" + (f", Claude Opus 5 {second[sample_id]}" if second else ""))
    curated = extract_saved_excerpt(REPO / task["gold_e_text_path"]) if task.get("gold_e_text_path") else None
    if curated:
        print(f"held-out passage ({task['gold_e_text_path']}): {norm(curated)[:width * 2]}")
    for index, step in enumerate(record["agent_trace"]):
        action, output = step["action"], step["output"]
        where = f"agent_trace[{index}]"
        if action == "FrameworkAgent":
            act = output.get("action", {})
            print(f"{where} orchestrator -> {act.get('agent') or act.get('action')}: {norm(act.get('reason'))[:width + 60]}")
        elif action in ("T-Inducer", "M-Inducer"):
            parsed = output.get("parsed", output)
            print(f"{where} {action}: {norm(json.dumps(parsed.get('outputs'), ensure_ascii=False))[:width * 2]}")
        elif action == "E.text-Finder":
            pool = output.get("pool", [])
            print(f"{where} E.text-Finder: {len(output.get('chunk_results', []))} chunk call(s), pool of {len(pool)}")
            for k, chunk in enumerate(output.get("chunk_results", [])):
                proposed = candidates_of(chunk)
                note = f" ERROR {norm(chunk['error'])[:90]}" if chunk.get("error") else ""
                print(f"    chunk_results[{k}] {chunk['chunk_id']}: {len(proposed)} candidate(s){note}")
            for k, candidate in enumerate(pool, 1):
                mark = "  <== held-out passage" if curated and same_place(candidate["excerpt"], curated) else ""
                print(f"    pool {k:2d} (confidence {candidate.get('confidence')}){mark}: {norm(candidate['excerpt'])[:width]}")
        elif action == "E.text-Reducer":
            reply = output.get("raw_output")
            print(f"{where} selector over {output.get('pool_size')}: " + (norm(reply) if reply else f"no reply parsed ({norm(output.get('error'))}); runtime falls back to confidence order"))
    print(f"returned: {norm(returned_excerpt(record))[:width * 3]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="gpt-oss-high", choices=MODELS)
    parser.add_argument("--write", action="store_true", help="write stage_attribution_<model>.csv next to this script")
    parser.add_argument("--show", metavar="SAMPLE_ID", help="print one execution step by step")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # consoles that cannot print every character of a report
    if args.show:
        show(args.model, args.show)
        return 0
    rows = attribute(args.model)
    print(f"{args.model}: Q2 executions against the held-out held-out passages"
          + ("" if SECOND_JUDGE else " (second judge's records not found; GPT-6.1 Terra scores only)"))
    print(summary(rows))
    if args.write:
        target = HERE / f"stage_attribution_{args.model}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
