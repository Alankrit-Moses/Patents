"""Shared tabular extraction utilities for Pattern 2."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def build_divergence_pair(
    *,
    first_entity: str,
    second_entity: str,
    start_year: int,
    first_start_count: int,
    second_start_count: int,
    end_year: int,
    first_end_count: int,
    second_end_count: int,
) -> list[dict[str, Any]]:
    """Build a two-row table showing that the pairwise gap widens."""
    start_gap = abs(first_start_count - second_start_count)
    end_gap = abs(first_end_count - second_end_count)
    gap_delta = end_gap - start_gap
    return [
        {
            "entity": first_entity,
            "pair_entity": second_entity,
            "start_year": start_year,
            "start_count": first_start_count,
            "end_year": end_year,
            "end_count": first_end_count,
            "start_gap": start_gap,
            "end_gap": end_gap,
            "gap_delta": gap_delta,
        },
        {
            "entity": second_entity,
            "pair_entity": first_entity,
            "start_year": start_year,
            "start_count": second_start_count,
            "end_year": end_year,
            "end_count": second_end_count,
            "start_gap": start_gap,
            "end_gap": end_gap,
            "gap_delta": gap_delta,
        },
    ]


def emit_result(result: dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        rows = result["tabular_example"]
        if not rows:
            raise ValueError("Cannot write an empty comparison table.")
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Saved {output_path}")
