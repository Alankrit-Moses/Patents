from __future__ import annotations

import difflib
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


# Length-preserving canonicalization: case-fold and map common unicode
# punctuation to ASCII so the verbatim check is not brittle to cosmetic
# differences (smart quotes, unicode dashes, capitalization, non-breaking
# spaces). Every replacement is 1 char -> 1 char so indices stay aligned and the
# ORIGINAL source span can be recovered.
_CANON_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ", " ": " ", " ": " ",
})


def _canon(text: str) -> str:
    return text.translate(_CANON_MAP).lower()


def recover_span_canonical(candidate: str, source: str) -> str | None:
    """Recover the real source span for a candidate, tolerating cosmetic
    differences (case, unicode punctuation/whitespace). Returns the ORIGINAL
    source text so downstream scoring sees the real report passage.
    """
    strict = recover_exact_span(candidate, source)
    if strict is not None:
        return strict
    candidate = candidate.strip()
    if not candidate:
        return None
    canon_source = _canon(source)
    if len(canon_source) != len(source):  # non-length-preserving fold; bail safely
        return None
    tokens = _canon(candidate).split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, canon_source, flags=re.DOTALL)
    return source[match.start():match.end()] if match else None


_SNAP_SM: dict = {}


def _sm_for(source_norm: str) -> difflib.SequenceMatcher:
    key = (len(source_norm), source_norm[:60], source_norm[-60:])
    sm = _SNAP_SM.get(key)
    if sm is None:
        sm = difflib.SequenceMatcher(autojunk=False)
        sm.set_seq2(source_norm)
        _SNAP_SM[key] = sm
    return sm


def snap_to_source(candidate: str, source: str, threshold: float = 0.8) -> str | None:
    """Fuzzy-snap a non-verbatim candidate to the real report passage it best
    matches. Grounding score = longest contiguous matching block / candidate
    length (contiguous, so scattered numeric/entity overlap from a table dump is
    rejected). Returns the matched ORIGINAL report window (whitespace-normalized)
    if it clears ``threshold``, else None. This decouples "found the passage"
    from "copied it exactly" without crediting hallucinations.
    """
    cand = " ".join(str(candidate).split()).strip()
    if not cand:
        return None
    src = " ".join(source.split())
    sm = _sm_for(src)
    sm.set_seq1(cand)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None
    if max(b.size for b in blocks) / len(cand) < threshold:
        return None
    start = min(b.b for b in blocks)
    end = max(b.b + b.size for b in blocks)
    if end - start > 3 * len(cand):  # guard against a stray distant match
        longest = max(blocks, key=lambda b: b.size)
        start, end = longest.b, min(len(src), longest.b + len(cand))
    return src[start:end]
