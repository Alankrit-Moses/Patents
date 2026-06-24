# Pattern 1 tabular-example extractors

These Python scripts reconstruct the tabular time series corresponding to each saved `E.text` instance. They print JSON unless `--output` is supplied. An output path ending in `.csv` saves only the readable two-column table; other extensions save the complete JSON record and metadata.

Requirements:

```text
Python 3.10+
openpyxl
```

Example:

```powershell
python Patterns/P1/E_tab_code/genAI/P1-E1_GAN_slowdown_after_2020.py
```

To save the extracted JSON later:

```powershell
python Patterns/P1/E_tab_code/genAI/P1-E1_GAN_slowdown_after_2020.py --output path/to/E_tab.csv
```

Add `--include-family-ids` when row-level family identifiers are needed for auditing. By default, each script prints only the annual distinct-family counts.
