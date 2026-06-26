"""Extract growth divergence for China and the United Kingdom in GenAI, 2014-2023."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import build_divergence_pair, emit_result


def extract() -> dict:
    rows = build_divergence_pair(
        first_entity="China",
        second_entity="United Kingdom",
        start_year=2014,
        first_start_count=278,
        second_start_count=35,
        end_year=2023,
        first_end_count=10591,
        second_end_count=107,
    )
    return {"instance_id": "P2-E2", "metric": "gap_delta", "tabular_example": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    emit_result(extract(), args.output)
