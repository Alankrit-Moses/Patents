# Pattern 2 CAGR extractors

Each Python script reconstructs a comparative-growth `E.tab` table from the corresponding workbook. The comparison metric is CAGR calculated from distinct patent-family counts over one shared interval.

Requirements:

```text
Python 3.10+
openpyxl
```

Run without `--output` to print JSON. Supply a `.csv` output path to save the readable comparison table.

Example:

```powershell
python Patterns/P2/E_tab_code/genAI/P2-E2_Inventor_country_growth_comparison.py --output Patterns/P2/E_tab/genAI/P2-E2_Inventor_country_growth_comparison.csv
```
