from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .io_utils import read_text, relative_path, write_jsonl


EXAMPLE_ID_RE = re.compile(r"^(P\d+-E\d+)", re.IGNORECASE)

EXP1_REPORTS_BY_PATTERN = {
    "P1": ("decarbonizing", "genAI", "ilmenite"),
    "P2": ("decarbonizing", "genAI", "OHS"),
}


@dataclass
class ExampleRecord:
    example_id: str
    pattern_id: str
    report_id: str
    e_text_path: str | None = None
    e_tab_path: str | None = None

    @property
    def paired(self) -> bool:
        return bool(self.e_text_path and self.e_tab_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "pattern_id": self.pattern_id,
            "report_id": self.report_id,
            "e_text_path": self.e_text_path,
            "e_tab_path": self.e_tab_path,
            "paired": self.paired,
        }


@dataclass
class PatternRecord:
    pattern_id: str
    m_path: str
    t_path: str
    examples: list[ExampleRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "m_path": self.m_path,
            "t_path": self.t_path,
            "examples": [example.to_dict() for example in self.examples],
        }


@dataclass
class Manifest:
    patterns: list[PatternRecord]
    reports: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patterns": [pattern.to_dict() for pattern in self.patterns],
            "reports": self.reports,
            "warnings": self.warnings,
        }


def _example_id(path: Path) -> str | None:
    match = EXAMPLE_ID_RE.match(path.stem)
    return match.group(1).upper() if match else None


def _scan_example_tree(root: Path, pattern_id: str, component: str) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    suffix = ".txt" if component == "E_text" else ".csv"
    base = root / "Patterns" / pattern_id / component
    if not base.exists():
        return found
    for path in sorted(base.rglob(f"*{suffix}")):
        example_id = _example_id(path)
        if not example_id:
            continue
        report_id = path.parent.name
        found[(report_id.casefold(), example_id)] = path
    return found


def build_manifest(config: HarnessConfig) -> Manifest:
    root = config.project_root
    warnings: list[str] = []
    report_paths = sorted((root / "reports" / "text versions").glob("*.txt"))
    reports_by_key = {path.stem.casefold(): path for path in report_paths}
    reports = {path.stem: relative_path(path, root) for path in report_paths}
    patterns: list[PatternRecord] = []

    for pattern_dir in sorted((root / "Patterns").glob("P*")):
        if not pattern_dir.is_dir():
            continue
        pattern_id = pattern_dir.name
        m_path = pattern_dir / "M.txt"
        t_path = pattern_dir / "T.txt"
        if not m_path.exists() or not t_path.exists():
            warnings.append(f"Skipping {pattern_id}: missing M.txt or T.txt")
            continue
        text_files = _scan_example_tree(root, pattern_id, "E_text")
        tab_files = _scan_example_tree(root, pattern_id, "E_tab")
        keys = sorted(set(text_files) | set(tab_files))
        examples: list[ExampleRecord] = []
        for report_key, example_id in keys:
            text_path = text_files.get((report_key, example_id))
            tab_path = tab_files.get((report_key, example_id))
            report_path = reports_by_key.get(report_key)
            report_id = report_path.stem if report_path else report_key
            if not report_path:
                warnings.append(
                    f"{pattern_id}/{example_id}: no report text matches folder '{report_key}'"
                )
            if text_path is None:
                warnings.append(f"{pattern_id}/{example_id}: unmatched E_tab")
            if tab_path is None:
                warnings.append(f"{pattern_id}/{example_id}: unmatched E_text")
            examples.append(
                ExampleRecord(
                    example_id=example_id,
                    pattern_id=pattern_id,
                    report_id=report_id,
                    e_text_path=relative_path(text_path, root) if text_path else None,
                    e_tab_path=relative_path(tab_path, root) if tab_path else None,
                )
            )
        patterns.append(
            PatternRecord(
                pattern_id=pattern_id,
                m_path=relative_path(m_path, root),
                t_path=relative_path(t_path, root),
                examples=examples,
            )
        )
    return Manifest(patterns=patterns, reports=reports, warnings=warnings)


def extract_saved_excerpts(path: Path) -> list[str]:
    text = read_text(path).strip()
    blocks = re.split(r"\r?\n\s*\r?\n", text, maxsplit=1)
    body = blocks[1].strip() if len(blocks) == 2 else text
    label_pattern = re.compile(
        r"(?im)^(?:[^\r\n:]{1,80}\s+)?excerpt:\s*$"
    )
    labels = list(label_pattern.finditer(body))
    if not labels:
        return [body]
    excerpts: list[str] = []
    for index, label in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(body)
        value = body[label.end() : end].strip()
        value = re.split(r"(?im)^Note:\s*", value, maxsplit=1)[0].strip()
        if value:
            excerpts.append(value)
    return excerpts


def extract_saved_excerpt(path: Path) -> str:
    return "\n\n".join(extract_saved_excerpts(path))


def _task_base(experiment: str, pattern: PatternRecord, task_id: str, query: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "experiment": experiment,
        "pattern_id": pattern.pattern_id,
        "framework_query": query,
        "m_path": pattern.m_path,
        "t_path": pattern.t_path,
    }


def build_tasks(manifest: Manifest, config: HarnessConfig) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    report_lookup = {name.casefold(): path for name, path in manifest.reports.items()}

    for pattern in manifest.patterns:
        exp1_reports = EXP1_REPORTS_BY_PATTERN.get(
            pattern.pattern_id,
            tuple(sorted(manifest.reports, key=str.casefold)),
        )
        for report_id in exp1_reports:
            report_path = report_lookup.get(report_id.casefold())
            if not report_path:
                continue
            task = _task_base(
                "1",
                pattern,
                f"exp1-{pattern.pattern_id}-{report_id}",
                "D:(D.text,_)\nP:(M,_,{(?,_)[3]})",
            )
            task.update({"report_id": report_id, "report_path": report_path, "target_count": 3})
            tasks.append(task)

        paired = [example for example in pattern.examples if example.paired]
        for example in paired:
            report_path = report_lookup.get(example.report_id.casefold())
            if not report_path:
                continue
            task = _task_base(
                "2",
                pattern,
                f"exp2-{example.example_id}",
                "D:(D.text,_)\nP:(_,_,{(?,E.tab)})",
            )
            task.update(
                {
                    "report_id": example.report_id,
                    "report_path": report_path,
                    "source_example_id": example.example_id,
                    "e_tab_path": example.e_tab_path,
                    "gold_e_text_path": example.e_text_path,
                    "target_count": 1,
                }
            )
            tasks.append(task)

        if len(paired) >= 2:
            induction_size = min(config.induction_pairs, len(paired) - 1)
            holdout_size = len(paired) - induction_size
            fold_total = min(config.fold_count, len(paired))
            for fold in range(fold_total):
                start = (fold * holdout_size) % len(paired)
                heldout_indices = {(start + index) % len(paired) for index in range(holdout_size)}
                heldout = [example for index, example in enumerate(paired) if index in heldout_indices]
                induction = [example for index, example in enumerate(paired) if index not in heldout_indices]
                task = _task_base(
                    "3",
                    pattern,
                    f"exp3-{pattern.pattern_id}-fold{fold + 1}",
                    "D:(_,_)\nP:(?,?,{(E1.text,E1.tab),(E2.text,E2.tab),...})",
                )
                task.update(
                    {
                        "fold": fold + 1,
                        "induction_examples": [example.to_dict() for example in induction],
                        "heldout_examples": [example.to_dict() for example in heldout],
                        "target_count": 1,
                    }
                )
                tasks.append(task)
    return tasks


def save_manifest_and_tasks(
    manifest: Manifest, tasks: list[dict[str, Any]], output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    import json

    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tasks_path = output_dir / "tasks.jsonl"
    write_jsonl(tasks_path, tasks)
    return manifest_path, tasks_path


def manifest_counts(manifest: Manifest, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    examples = [example for pattern in manifest.patterns for example in pattern.examples]
    return {
        "patterns": len(manifest.patterns),
        "text_examples": sum(bool(example.e_text_path) for example in examples),
        "tabular_examples": sum(bool(example.e_tab_path) for example in examples),
        "paired_examples": sum(example.paired for example in examples),
        "tasks": {
            experiment: sum(task["experiment"] == experiment for task in tasks)
            for experiment in ("1", "2", "3")
        },
        "warnings": manifest.warnings,
    }
