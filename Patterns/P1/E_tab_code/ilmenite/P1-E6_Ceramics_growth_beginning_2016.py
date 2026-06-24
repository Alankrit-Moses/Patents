"""Extract E.tab for P1-E6: growth in titanium use for ceramics beginning in 2016."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import emit_json, extract_distinct_family_counts, find_repository_root


def extract(include_family_ids: bool = False) -> dict:
    root = find_repository_root(Path(__file__))
    workbook = root / "tables" / "ilmenite" / "wipo-pub-1077-23-tech1.xlsx"
    data = extract_distinct_family_counts(
        workbook_path=workbook,
        sheet_name="Ceramics",
        header_row=3,
        family_column="Questel unique family ID (FAN)",
        year_column="Earliest priority date",
        start_year=2012,
        end_year=2019,
        count_field="distinct_titanium_ceramics_patent_family_count",
        include_family_ids=include_family_ids,
    )
    return {
        "instance_id": "P1-E6",
        "pattern": "Trajectory Shift",
        "entity": "Patent families describing titanium or titanium dioxide use in ceramics",
        "attribute": "annual distinct active patent-family count by earliest priority year",
        "tau": 2016,
        "source_workbook": str(workbook.relative_to(root)),
        "source_sheet": "Ceramics",
        "filters": {},
        "aggregation": "COUNT(DISTINCT Questel unique family ID (FAN)) BY earliest priority year",
        "analysis_window": "2012-2019; 2020-2022 omitted because the report identifies them as incomplete",
        "tabular_example": data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--include-family-ids", action="store_true")
    args = parser.parse_args()
    emit_json(extract(args.include_family_ids), args.output)
