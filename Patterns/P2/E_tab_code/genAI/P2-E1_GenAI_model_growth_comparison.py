"""Extract CAGR comparison for GenAI model families, 2020-2023."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import emit_result, extract_cagr_comparison, find_repository_root


def extract() -> dict:
    root = find_repository_root(Path(__file__))
    workbook = root / "tables" / "genAI" / "wipo-pub-2007-tech1.xlsx"
    names = [
        "GAN, Generative Adversarial Networks",
        "Variational Autoencoder,VAE",
        "Autoregressiv Models",
        "Diffusion Models",
        "Large Language Models,LLM",
    ]
    rows = extract_cagr_comparison(
        workbook_path=workbook,
        start_year=2020,
        end_year=2023,
        sheet_specs=[{
            "sheet_name": "GenAI total, models and modes",
            "header_row": 1,
            "family_column": "PATENT_FAMILY_ID",
            "year_column": "PUBLICATION_YEAR",
            "entities": [
                {"entity": name, "filters": {"TECHNOLOGY_NAME": name}} for name in names
            ],
        }],
    )
    return {"instance_id": "P2-E1", "metric": "CAGR", "tabular_example": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    emit_result(extract(), args.output)
