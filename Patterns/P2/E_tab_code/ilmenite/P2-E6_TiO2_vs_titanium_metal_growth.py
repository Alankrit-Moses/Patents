"""Extract CAGR comparison for titanium dioxide and titanium metal production, 2002-2019."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import emit_result, extract_cagr_comparison, find_repository_root


def extract() -> dict:
    root = find_repository_root(Path(__file__))
    workbook = root / "tables" / "ilmenite" / "wipo-pub-1077-23-tech1.xlsx"
    base = {
        "header_row": 3,
        "family_column": "Questel unique family ID (FAN)",
        "year_column": "Earliest priority date",
    }
    rows = extract_cagr_comparison(
        workbook_path=workbook,
        start_year=2002,
        end_year=2019,
        sheet_specs=[
            {
                **base,
                "sheet_name": "Titanium dioxide",
                "entities": [{"entity": "Titanium dioxide production", "filters": {}}],
            },
            {
                **base,
                "sheet_name": "Titanium metal",
                "entities": [{"entity": "Titanium metal production", "filters": {}}],
            },
        ],
    )
    return {"instance_id": "P2-E6", "metric": "CAGR", "tabular_example": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    emit_result(extract(), args.output)
