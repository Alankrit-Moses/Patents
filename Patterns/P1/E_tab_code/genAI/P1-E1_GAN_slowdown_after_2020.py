"""Extract E.tab for P1-E1: GAN slowdown after 2020."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import emit_json, extract_distinct_family_counts, find_repository_root


def extract(include_family_ids: bool = False) -> dict:
    root = find_repository_root(Path(__file__))
    workbook = root / "tables" / "genAI" / "wipo-pub-2007-tech1.xlsx"
    filters = {"TECHNOLOGY_NAME": "GAN, Generative Adversarial Networks"}
    data = extract_distinct_family_counts(
        workbook_path=workbook,
        sheet_name="GenAI total, models and modes",
        header_row=1,
        family_column="PATENT_FAMILY_ID",
        year_column="PUBLICATION_YEAR",
        start_year=2017,
        end_year=2023,
        filters=filters,
        count_field="distinct_gan_patent_family_count",
        include_family_ids=include_family_ids,
    )
    return {
        "instance_id": "P1-E1",
        "pattern": "Trajectory Shift",
        "entity": "GAN patent families",
        "attribute": "annual distinct published patent-family count",
        "tau": 2020,
        "source_workbook": str(workbook.relative_to(root)),
        "source_sheet": "GenAI total, models and modes",
        "filters": filters,
        "aggregation": "COUNT(DISTINCT PATENT_FAMILY_ID) BY PUBLICATION_YEAR",
        "tabular_example": data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--include-family-ids", action="store_true")
    args = parser.parse_args()
    emit_json(extract(args.include_family_ids), args.output)
