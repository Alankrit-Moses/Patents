"""Extract growth divergence for titanium dioxide and titanium metal production, 2002-2019."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import build_divergence_pair, emit_result


def extract() -> dict:
    rows = build_divergence_pair(
        first_entity="Titanium dioxide production",
        second_entity="Titanium metal production",
        start_year=2002,
        first_start_count=9,
        second_start_count=6,
        end_year=2019,
        first_end_count=40,
        second_end_count=7,
    )
    return {"instance_id": "P2-E6", "metric": "gap_delta", "tabular_example": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    emit_result(extract(), args.output)
