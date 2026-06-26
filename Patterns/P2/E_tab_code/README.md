# Pattern 2 growth-divergence extractors

Each Python script reconstructs a growth-divergence `E.tab` table for one example pair. The comparison metric is `gap_delta`, calculated as:

```text
gap_delta = |end_count(entity_a) - end_count(entity_b)| - |start_count(entity_a) - start_count(entity_b)|
```

A positive `gap_delta` confirms that the two time series moved farther apart over the shared interval.

Requirements:

```text
Python 3.10+
```

Run without `--output` to print JSON. Supply a `.csv` output path to save the readable comparison table.

Example:

```powershell
python Patterns/P2/E_tab_code/genAI/P2-E2_Inventor_country_growth_comparison.py --output Patterns/P2/E_tab/genAI/P2-E2_Inventor_country_growth_comparison.csv
```
