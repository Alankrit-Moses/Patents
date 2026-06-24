from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_GENERATOR_PARTS = {"e_tab_code", "tables"}
FORBIDDEN_GENERATOR_SUFFIXES = {".xlsx", ".xls", ".xlsm"}


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Unable to decode {path}")


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_safe_input(root: Path, relative: str, allowed_suffixes: set[str]) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    lowered_parts = {part.casefold() for part in path.parts}
    if lowered_parts & FORBIDDEN_GENERATOR_PARTS:
        raise ValueError(f"Forbidden generator input path: {relative}")
    if path.suffix.casefold() in FORBIDDEN_GENERATOR_SUFFIXES:
        raise ValueError(f"Raw workbook input is forbidden: {relative}")
    if path.suffix.casefold() not in allowed_suffixes:
        raise ValueError(f"Unexpected input type for {relative}")
    return path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        starts = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0]
        if not starts:
            raise
        start = min(starts)
        opener = stripped[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(stripped[start : index + 1])
        raise json.JSONDecodeError("No complete JSON value", stripped, start)


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
