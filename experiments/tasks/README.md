# Released study tasks

- `manifest.json` inventories the two pattern cards, their curated examples,
  and the report text available to task construction.
- `tasks.jsonl` contains the 24 concrete tasks evaluated in the paper: six Q1,
  twelve Q2, and six Q3 tasks.

These files can be regenerated with:

```powershell
python -m experiments.cli --config experiments/configs/gpt-4.1.json manifest
```
