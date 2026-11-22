# Suggested task input assignment

These are **task mappings**, not model responses. The six gold `e.text` passages must be held out in Q2, as usual.

## Q1 — Example mining (three task inputs)

- **Q1-A:** `D.text/Q1_A_aviation.txt` + M; at least three matching passages: E1, E2, E3.
- **Q1-B:** `D.text/Q1_B_software.txt` + M; at least three matching passages: E4, E5, E6.
- **Q1-C:** `D.text/Q1_C_combined.txt` + M; at least six matching passages E1-E6, any three eligible.

## Q2 — Example alignment (six inputs)

For each Ei use the paired `examples/Ei.e_tab.csv`, and the following D.text source corpus. Withhold its `Ei.e_text.txt` from the generator and judge; judge the proposed passage directly against the table.

- E1, E2, E3 → `D.text/sources/aviation_ao_2021_053.txt`
- E4 → `D.text/sources/cloudflare_2025_06_12.txt`
- E5, E6 → `D.text/sources/cloudflare_2025_09_12.txt`

## Q3 — Pattern induction (three distinct four-pair subsets)

- **Q3-A:** E1, E2, E4, E5
- **Q3-B:** E1, E3, E4, E6
- **Q3-C:** E2, E3, E5, E6

Each Q3 input contains the full four respective `(e.text,e.tab)` pairs; the target outputs are M and T. Every subset contains two aviation and two software cases.
