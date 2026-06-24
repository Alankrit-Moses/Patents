from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from .io_utils import read_text


@dataclass(frozen=True)
class ReportChunk:
    chunk_id: str
    start_char: int
    end_char: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_report(path: Path) -> str:
    return read_text(path)


def chunk_report(text: str, chunk_chars: int, overlap_chars: int) -> list[ReportChunk]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be in [0, chunk_chars)")
    chunks: list[ReportChunk] = []
    start = 0
    index = 0
    while start < len(text):
        desired_end = min(len(text), start + chunk_chars)
        end = desired_end
        if desired_end < len(text):
            boundary = text.rfind("\n\n", start + chunk_chars // 2, desired_end)
            if boundary > start:
                end = boundary + 2
        chunks.append(ReportChunk(f"chunk-{index:04d}", start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
        index += 1
    return chunks


def recover_exact_span(candidate: str, source: str) -> str | None:
    candidate = candidate.strip()
    if not candidate:
        return None
    if candidate in source:
        return candidate
    tokens = candidate.split()
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, source, flags=re.DOTALL)
    return match.group(0) if match else None
