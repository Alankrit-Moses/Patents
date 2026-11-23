#!/usr/bin/env python3
"""Re-judge the released generator outputs with a Claude judge.

Reused unchanged from the released code (experiments/evaluation.py): the judge
system message, every rubric prompt, the grounding rule, the score parsing and
the record format. Only the model client differs.

The judge is given those prompts and never an earlier score: this process opens
the generator outputs and the task inputs they reference, and cannot open an
earlier evaluation, summary or results table (see _guard).

Two ways to reach the judge model:
  --backend claude-code   Claude Code's non-interactive mode (`claude -p`), using
                          the local Claude Code login. No API key.
  --backend api           the Anthropic API (ANTHROPIC_API_KEY, or one line
                          ANTHROPIC_API_KEY=... in keys.env next to this file).

Every judge call is appended to results/<judge model>/calls.jsonl with its
prompt, raw reply and token usage. A rerun reuses the logged calls, so an
interrupted run resumes without judging anything twice, and the evaluation
files are a pure function of that log.

  python3 -I run_opus_judge.py --dry-run                                   # no model calls
  python3 -I run_opus_judge.py --backend claude-code --model claude-opus-5 --limit 2   # pilot
  python3 -I run_opus_judge.py --backend claude-code --model claude-opus-5             # full run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _find_repo() -> Path:
    """The study repository: this folder's parent if it was placed inside it."""
    if os.environ.get("PATTERNS_REPO"):
        return Path(os.environ["PATTERNS_REPO"]).resolve()
    for candidate in (HERE.parent, HERE.parent / "repo_latest"):
        if (candidate / "experiments" / "evaluation.py").exists():
            return candidate.resolve()
    sys.exit("cannot find the study repository; set PATTERNS_REPO to its path")


REPO = _find_repo()

USD_PER_MTOK = {  # judge model: API list prices
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_write": 6.25, "cache_read": 0.50},
}
DEFAULT_JUDGE = "claude-opus-5"
GENERATORS = ["gpt-4.1", "gpt-oss-high", "gpt-oss-medium", "gpt-5.6-luna"]
SOURCES = {
    "p1-p2": lambda g: (REPO / "results" / g / "outputs.jsonl", REPO / "experiments/tasks/tasks.jsonl"),
    "p3": lambda g: (REPO / "results/final" / g / "p3-outputs.jsonl", REPO / "experiments/tasks/p3.jsonl"),
}
# API backend only. The released judge config asks for JSON mode
# (response_format json_object); this is the same constraint as a schema.
SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
    "additionalProperties": False,
}

_ALLOWED_RESULT_FILES = {"outputs.jsonl", "p3-outputs.jsonl"}
_SCORE_FILE_TAGS = ("evaluation", "summary", "results_table")
_GUARDED_DIRS = (str(REPO / "results") + os.sep, str(REPO / "experiments" / "runs") + os.sep)


def _guard(event: str, args: tuple) -> None:
    """Refuse to open any file that holds earlier judge scores."""
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        path = os.path.abspath(os.fsdecode(args[0]))
    except TypeError:
        return
    if path.startswith(str(HERE) + os.sep) or not path.startswith(str(REPO) + os.sep):
        return  # this tool's own files, or nothing to do with the study repository
    name = os.path.basename(path).lower()
    in_results = path.startswith(_GUARDED_DIRS) and os.path.isfile(path) and name not in _ALLOWED_RESULT_FILES
    tagged = name.endswith((".jsonl", ".json", ".csv")) and any(tag in name for tag in _SCORE_FILE_TAGS)
    if in_results or tagged:
        raise PermissionError(f"the judge run must not read earlier judge scores: {path}")


sys.addaudithook(_guard)

# Released judge configuration (results/final/README.md).
os.environ["GROUNDING_SNAP_THRESHOLD"] = "0.8"
os.environ["EVALUATION_SCALE"] = "5"
os.environ["Q3_M_RUBRIC"] = "moderate"

sys.path.append(str(REPO))
from experiments import evaluation as ev  # noqa: E402
from experiments.config import EndpointConfig, HarnessConfig  # noqa: E402
from experiments.io_utils import load_jsonl  # noqa: E402


class JudgeFailure(RuntimeError):
    """A judge call that produced no usable reply."""


@dataclass
class Reply:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


def claude_code_command(model: str, system: str, effort: str) -> list[str]:
    """One judge call through Claude Code: no tools, no MCP servers, no skills,
    no user or project settings, thinking off. The user prompt goes on stdin."""
    return [
        "claude", "-p", "--model", model, "--system-prompt", system,
        "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
        "--setting-sources", "", "--settings", json.dumps({"alwaysThinkingEnabled": False}),
        "--effort", effort, "--no-session-persistence", "--output-format", "json",
    ]


def is_bare_score(text: str) -> bool:
    """True if the reply is the JSON score object and nothing else, as the prompt requires."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and set(value) == {"score"}


def final_score_object(text: str) -> str | None:
    """The JSON score object a reply consists of or ends with.

    The prompt asks for that object alone. With thinking off the judge
    occasionally writes a rationale before it; the judgment is kept (one call
    per judgment, no repeats) and the score is the object the reply ends with.
    """
    found = re.findall(r'\{\s*"score"\s*:\s*-?\d+\s*\}', text)
    return found[-1] if found else None


class OpusJudge:
    """Makes and logs judge calls; one logged reply per (record, call position)."""

    def __init__(self, *, client: Any, model: str, mode: str, max_tokens: int, log_path: Path, dry_run: bool,
                 backend: str = "api", effort: str | None = None, workdir: str | None = None):
        self.client = client
        self.model = model
        self.mode = mode
        self.max_tokens = max_tokens
        self.log_path = log_path
        self.dry_run = dry_run
        self.backend = backend
        self.effort = effort
        self.workdir = workdir
        self.lock = threading.Lock()
        self.calls: dict[tuple[str, int], dict[str, Any]] = {}
        self.fatal: BaseException | None = None
        self.new_calls = 0
        self.reused = 0
        self.dry_chars: list[int] = []
        if not dry_run and log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.calls[(entry["key"], entry["ordinal"])] = entry

    def _abort(self, reason: str) -> JudgeFailure:
        self.fatal = RuntimeError(reason)
        return JudgeFailure(reason)

    def request(self, system: str, user: str) -> dict[str, Any]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": SCORE_SCHEMA}}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.mode == "disabled":
            kwargs["thinking"] = {"type": "disabled"}
        else:
            output_config["effort"] = self.mode
        kwargs["output_config"] = output_config
        return kwargs

    def _via_api(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        import anthropic

        try:
            message = self.client.messages.create(**self.request(system, user))
        except (
            anthropic.BadRequestError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
        ) as error:
            raise self._abort(f"{type(error).__name__}: {error}") from error
        except anthropic.RateLimitError as error:
            raise JudgeFailure(f"rate limited after retries: {error}") from error
        except anthropic.APIStatusError as error:
            raise JudgeFailure(f"API error {error.status_code}: {error}") from error
        except anthropic.APIConnectionError as error:
            raise JudgeFailure(f"connection error: {error}") from error

        if message.stop_reason == "refusal":
            category = getattr(getattr(message, "stop_details", None), "category", None)
            raise JudgeFailure(f"judge declined (stop_reason=refusal, category={category})")
        if message.stop_reason != "end_turn":
            raise JudgeFailure(f"unexpected stop_reason: {message.stop_reason}")
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, {
            "served_model": message.model,
            "stop_reason": message.stop_reason,
            "message_id": message.id,
            "request_id": getattr(message, "_request_id", None),
            "usage": message.usage.to_dict(),
        }

    def _via_claude_code(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        command = claude_code_command(self.model, system, self.effort or "high")
        try:
            done = subprocess.run(command, input=user, capture_output=True, text=True, encoding="utf-8",
                                  cwd=self.workdir, timeout=600)
        except (subprocess.TimeoutExpired, OSError) as error:
            raise self._abort(f"claude CLI did not complete (is Claude Code installed?): {error}") from error
        try:
            data = json.loads(done.stdout)
        except json.JSONDecodeError as error:
            raise self._abort(f"claude CLI gave no JSON (exit {done.returncode}): "
                              f"{(done.stderr or done.stdout)[:300]}") from error
        if data.get("stop_reason") == "refusal":
            raise JudgeFailure("judge declined (stop_reason=refusal): " + " ".join(str(data.get("result")).split())[:260])
        if data.get("is_error") or data.get("stop_reason") != "end_turn" or data.get("num_turns") != 1:
            raise self._abort(f"claude CLI error (status {data.get('api_error_status')}, stop {data.get('stop_reason')}, "
                              f"turns {data.get('num_turns')}): {' '.join(str(data.get('result')).split())[:300]}")
        served = sorted(data.get("modelUsage") or {})
        usage = data.get("usage") or {}
        thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if served != [self.model]:
            raise self._abort(f"call was served by {served}, not {self.model}")
        if thinking != 0:
            raise self._abort(f"reasoning was not off: {thinking} thinking tokens")
        return str(data.get("result") or ""), {
            "served_model": served[0],
            "stop_reason": data["stop_reason"],
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "thinking_tokens": thinking,
            },
            "list_price_usd": data.get("total_cost_usd"),
            "duration_ms": data.get("duration_ms"),
        }

    def call(self, key: str, ordinal: int, system: str, user: str) -> Reply:
        digest = hashlib.sha256((system + "\x00" + user).encode("utf-8")).hexdigest()
        with self.lock:
            entry = self.calls.get((key, ordinal))
        if entry is not None:
            logged = (entry["prompt_sha256"], entry["mode"], entry["requested_model"], entry.get("backend", "api"))
            if logged != (digest, self.mode, self.model, self.backend):
                raise JudgeFailure(f"logged call {key}#{ordinal} used a different prompt or setting")
            with self.lock:
                self.reused += 1
            return Reply(entry["score_json"], usage=entry["usage"])
        if self.dry_run:
            with self.lock:
                self.dry_chars.append(len(system) + len(user))
            return Reply('{"score": 3}')
        if self.fatal is not None:
            raise JudgeFailure(f"run aborted: {self.fatal}")

        send = self._via_claude_code if self.backend == "claude-code" else self._via_api
        text, meta = send(system, user)
        score_json = final_score_object(text)
        if score_json is None:
            raise JudgeFailure("reply contains no JSON score object")
        entry = {
            "key": key,
            "ordinal": ordinal,
            "prompt_sha256": digest,
            "text": text,
            "score_json": score_json,
            "requested_model": self.model,
            "backend": self.backend,
            "mode": self.mode,
            **meta,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "system": system,
            "user": user,
        }
        with self.lock:
            self.calls[(key, ordinal)] = entry
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.new_calls += 1
        return Reply(score_json, usage=entry["usage"])


class RecordClient:
    """The client evaluation.py talks to while judging one output record."""

    def __init__(self, judge: OpusJudge, key: str):
        self.judge = judge
        self.key = key
        self.ordinal = 0
        self.failure: BaseException | None = None

    def complete(self, messages: list[dict[str, str]], *, temperature: float | None = None,
                 max_tokens: int | None = None) -> Reply:
        # temperature is part of the released interface but is not sent:
        # current Claude Opus models reject sampling parameters.
        if [message["role"] for message in messages] != ["system", "user"]:
            raise ValueError("unexpected judge message layout")
        ordinal, self.ordinal = self.ordinal, self.ordinal + 1
        try:
            return self.judge.call(self.key, ordinal, messages[0]["content"], messages[1]["content"])
        except Exception as error:
            self.failure = error
            raise


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    key_file = HERE / "keys.env"
    if key_file.exists():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            value = value.strip().strip("'\"")
            if name.strip() == "ANTHROPIC_API_KEY" and value:
                os.environ["ANTHROPIC_API_KEY"] = value
                return
    sys.exit("No Anthropic credentials: set ANTHROPIC_API_KEY or put ANTHROPIC_API_KEY=... in "
             f"{key_file} (or use --backend claude-code)")


def thinking_can_be_disabled(client: Any, model: str) -> bool:
    """Ask the API itself whether this model can run with thinking off."""
    import anthropic

    try:
        client.messages.create(
            model=model,
            max_tokens=16,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
        )
    except anthropic.BadRequestError as error:
        if "thinking" in str(error).lower():
            return False
        raise
    return True


def pin_settings(out_dir: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Write the judge settings once; refuse to mix settings in one result set."""
    path = out_dir / "judge_settings.json"
    identity = ("judge_model", "backend", "thinking_mode", "effort")
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if any(recorded.get(name) != settings.get(name) for name in identity):
            sys.exit(f"{path} records {[recorded.get(n) for n in identity]}; "
                     f"refusing to mix with {[settings.get(n) for n in identity]}")
        return recorded
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    settings = {
        **settings,
        "thinking_mode_meaning": "'disabled' = reasoning (extended thinking) off; otherwise the effort level with thinking on",
        "sampling": "model default (this model rejects temperature/top_p/top_k)",
        "refusal_fallback": "none: every score comes from the judge model",
        "reply_format": "one call per judgment; the score is the JSON score object the reply consists of or ends with "
                        "(a reply that also carries a written rationale is kept, not repeated)",
        "grounding_snap_threshold": os.environ["GROUNDING_SNAP_THRESHOLD"],
        "evaluation_scale": os.environ["EVALUATION_SCALE"],
        "q3_m_rubric": os.environ["Q3_M_RUBRIC"],
        "prompts_from": f"github.com/Alankrit-Moses/Patents@{commit} experiments/evaluation.py",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return settings


def resolve_mode(requested: str, client: Any, model: str, out_dir: Path, max_tokens: int) -> str:
    """API backend: thinking off if the API allows it for this model, else effort low."""
    path = out_dir / "judge_settings.json"
    probe = None
    if path.exists():
        mode = json.loads(path.read_text(encoding="utf-8"))["thinking_mode"]
        if requested not in ("auto", mode):
            sys.exit(f"{path} records thinking mode {mode!r}; refusing to mix it with {requested!r}")
    elif requested == "auto":
        probe = thinking_can_be_disabled(client, model)
        mode = "disabled" if probe else "low"
    else:
        mode = requested
    import anthropic

    pin_settings(out_dir, {
        "judge_model": model,
        "backend": "api",
        "thinking_mode": mode,
        "effort": None if mode == "disabled" else mode,
        "api_accepts_thinking_disabled": probe,
        "max_tokens": max_tokens,
        "output_format": {"type": "json_schema", "schema": SCORE_SCHEMA},
        "anthropic_sdk": anthropic.__version__,
    })
    return mode


def judge_file(judge: OpusJudge, config: HarnessConfig, generator: str, kind: str,
               limit: int | None, concurrency: int):
    outputs_path, tasks_path = SOURCES[kind](generator)
    tasks = {task["task_id"]: task for task in load_jsonl(tasks_path)}
    results = load_jsonl(outputs_path)
    if not results or not tasks:
        sys.exit(f"missing generator outputs or tasks: {outputs_path} / {tasks_path}")
    if len({result["sample_id"] for result in results}) != len(results):
        sys.exit(f"sample ids are not unique in {outputs_path}")
    total = len(results)
    if limit:
        results = results[:limit]

    def work(result: dict[str, Any]):
        client = RecordClient(judge, f"{generator}/{kind}/{result['sample_id']}")
        return client, ev._judge_one(result, tasks, client, config)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        judged = list(pool.map(work, results))
    records, failures = [], []
    for result, (client, record) in zip(results, judged):
        if client.failure is not None or (record["errors"] and not result.get("errors")):
            failures.append((client.key, "; ".join(record["errors"]) or str(client.failure)))
        else:
            records.append(record)
    return records, failures, total


def count_scores(records: list[dict[str, Any]]) -> tuple[int, int]:
    scores: list[int] = []
    for record in records:
        for value in record["metrics"].values():
            scores.extend(value if isinstance(value, list) else [value])
    return len(scores), sum(1 for score in scores if score == 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="build every judge prompt, call nothing")
    parser.add_argument("--limit", type=int, help="pilot: judge only the first N records of each file")
    parser.add_argument("--generators", nargs="+", default=GENERATORS, choices=GENERATORS)
    parser.add_argument("--backend", default="claude-code", choices=["claude-code", "api"])
    parser.add_argument("--model", default=DEFAULT_JUDGE, choices=sorted(USD_PER_MTOK))
    parser.add_argument("--thinking", default="auto", choices=["auto", "disabled", "low", "medium", "high"],
                        help="api backend: auto = thinking off if the API allows it for this model, else effort low. "
                             "The claude-code backend always runs with thinking off.")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high"],
                        help="claude-code backend: effort level sent with thinking off (the API default is high)")
    parser.add_argument("--max-tokens", type=int, default=16000, help="api backend")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, help="default: results/<judge model> next to this file")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # consoles that cannot print every character of a reply
    out_dir = (args.out or HERE / "results" / args.model).resolve()
    prices = USD_PER_MTOK[args.model]

    client, effort, workdir = None, None, None
    if args.dry_run:
        mode, label = "dry-run", "DRY-RUN-STUB"
    elif args.backend == "claude-code":
        if args.thinking not in ("auto", "disabled"):
            sys.exit("the claude-code backend judges with thinking off only")
        mode, effort, label = "disabled", args.effort, args.model
        try:
            version = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
        except OSError:  # Claude Code is not installed; fine as long as every call is already in the log
            version = "not installed"
        pin_settings(out_dir, {
            "judge_model": args.model,
            "backend": "claude-code",
            "thinking_mode": mode,
            "effort": effort,
            "claude_code_version": version,
            "command": claude_code_command(args.model, "<judge system message>", effort) + ["<", "<judge user message>"],
            "note": "Claude Code appends its own environment block (date, platform, working directory, model name, "
                    "account identity, generic tool-safety notes) to every request; the judge system and user "
                    "messages are otherwise sent as logged in calls.jsonl.",
        })
        workdir = tempfile.mkdtemp(prefix="judge-")
    else:
        load_key()
        import anthropic

        client = anthropic.Anthropic(max_retries=8)
        try:
            mode = resolve_mode(args.thinking, client, args.model, out_dir, args.max_tokens)
        except anthropic.APIError as error:
            sys.exit(f"could not reach {args.model}: {type(error).__name__}: {error}")
        label = args.model
    if not args.dry_run:
        print(f"judge {args.model} via {args.backend}, thinking mode: {mode}" + (f", effort {effort}" if effort else ""),
              flush=True)
    judge = OpusJudge(client=client, model=args.model, mode=mode, max_tokens=args.max_tokens,
                      log_path=out_dir / "calls.jsonl", dry_run=args.dry_run,
                      backend=args.backend, effort=effort, workdir=workdir)
    config = HarnessConfig(project_root=REPO, judge=EndpointConfig(model=label))

    all_failures: list[tuple[str, str]] = []
    try:
        for generator in args.generators:
            for kind in SOURCES:
                before = judge.new_calls + judge.reused + len(judge.dry_chars)
                records, failures, total = judge_file(judge, config, generator, kind, args.limit, args.concurrency)
                calls = judge.new_calls + judge.reused + len(judge.dry_chars) - before
                n_scores, n_zero = count_scores(records)
                print(f"{generator:15s} {kind:6s} records {len(records)}/{total}  scores {n_scores}  "
                      f"judge calls {calls}  zeros assigned without a call {n_zero}  failures {len(failures)}", flush=True)
                all_failures.extend(failures)
                complete = not failures and len(records) == total
                if complete and (not args.dry_run or args.out is not None):
                    target = out_dir / generator / f"{kind}-evaluations.jsonl"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
                if judge.fatal is not None:
                    break
            if judge.fatal is not None:
                break
    finally:
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    if args.dry_run:
        chars = sum(judge.dry_chars)
        low, high = chars / 4.0, chars / 2.8
        print(f"\njudge calls: {len(judge.dry_chars)}   prompt characters: {chars:,}")
        print(f"input tokens (estimate): {low:,.0f} to {high:,.0f}")
        for name, out_tokens in (("no thinking, ~12 output tokens per call", 12), ("~400 thinking tokens per call", 412)):
            cost_low = (low * prices["input"] + len(judge.dry_chars) * out_tokens * prices["output"]) / 1e6
            cost_high = (high * prices["input"] + len(judge.dry_chars) * out_tokens * prices["output"]) / 1e6
            print(f"API cost estimate on {args.model}, {name}: ${cost_low:.2f} to ${cost_high:.2f}")
        return 0

    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    for entry in judge.calls.values():
        for name in usage:
            usage[name] += entry["usage"].get(name) or 0
    if args.backend == "claude-code":
        cost = sum(entry.get("list_price_usd") or 0 for entry in judge.calls.values())
    else:
        cost = (usage["input_tokens"] * prices["input"] + usage["output_tokens"] * prices["output"]
                + usage["cache_creation_input_tokens"] * prices["cache_write"]
                + usage["cache_read_input_tokens"] * prices["cache_read"]) / 1e6
    served = sorted({entry["served_model"] for entry in judge.calls.values()})
    print(f"\ncalls logged: {len(judge.calls)} (new this run {judge.new_calls}, reused {judge.reused})  served by: {served}")
    wordy = sum(1 for entry in judge.calls.values() if not is_bare_score(entry["text"]))
    print(f"replies carrying text besides the score object: {wordy} of {len(judge.calls)}")
    print(f"tokens: {usage}  at list prices: ${cost:.2f}")
    for key, reason in all_failures:
        print(f"FAILED {key}: {reason}")
    if judge.fatal is not None:
        print(f"ABORTED: {judge.fatal}")
    return 1 if all_failures or judge.fatal is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
