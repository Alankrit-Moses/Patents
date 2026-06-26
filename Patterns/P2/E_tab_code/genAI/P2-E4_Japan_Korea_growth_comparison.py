"""Extract growth divergence for Korea and Japan in GenAI, 2021-2023."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import build_divergence_pair, emit_result


def extract() -> dict:
    rows = build_divergence_pair(
        first_entity="Republic of Korea",
        second_entity="Japan",
        start_year=2021,
        first_start_count=815,
        second_start_count=549,
        end_year=2023,
        first_end_count=1054,
        second_end_count=399,
    )
    return {"instance_id": "P2-E4", "metric": "gap_delta", "tabular_example": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    emit_result(extract(), args.output)
