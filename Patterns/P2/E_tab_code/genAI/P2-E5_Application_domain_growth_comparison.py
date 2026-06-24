"""Extract CAGR comparison for selected GenAI application domains, 2021-2023."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import emit_result, extract_cagr_comparison, find_repository_root


def extract() -> dict:
    root = find_repository_root(Path(__file__))
    workbook = root / "tables" / "genAI" / "wipo-pub-2007-tech1.xlsx"
    applications = [
        ("Energy management", "EnergyManagement_2010-2023"),
        ("Agriculture", "Agriculture_2010-2024"),
        ("Life sciences", "Life and medical sciences"),
        ("Security", "Security"),
        ("Physical sciences and engineering", "Physical sciences and engineering"),
        ("Telecommunications", "Telecommunications_2010-2023"),
        ("Military", "Military"),
        ("Arts and humanities", "Arts and humanities"),
        (
            "Industrial property, law, social and behavioral sciences",
            "Industrial Property, Law, social and behavioral sciences",
        ),
    ]
    rows = extract_cagr_comparison(
        workbook_path=workbook,
        start_year=2021,
        end_year=2023,
        sheet_specs=[{
            "sheet_name": "GenAI applications",
            "header_row": 1,
            "family_column": "PATENT_FAMILY_ID",
            "year_column": "PUBLICATION_YEAR",
            "entities": [
                {"entity": label, "filters": {"TECHNOLOGY_NAME": workbook_label}}
                for label, workbook_label in applications
            ],
        }],
    )
    return {"instance_id": "P2-E5", "metric": "CAGR", "tabular_example": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    emit_result(extract(), args.output)
