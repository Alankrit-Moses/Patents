# P3 — Verbatim dependency-chain benchmark (6 pairs; 3 source reports)

**Scope.** Three aviation examples from ATSB AO-2021-053; three software examples from two Cloudflare incident postmortems. All three source files were directly supplied by the user and processed locally.

## Main files

- `M_and_T.md`: mathematical definition and textual description of the single symbolic pattern P3.
- `examples/E1.e_text.txt` ... `examples/E6.e_text.txt`: original contiguous report passages, **not paraphrases**. PDF line wraps are retained; HTML text represents publisher wording with HTML whitespace collapsed for readability.
- `examples/E1.e_tab.csv` ... `examples/E6.e_tab.csv`: actual report event rows and/or source-supported observations for the same chain. Event descriptions are not invented. Untimed observations have empty time fields rather than made-up timestamps.
- `D.text/sources/*.txt`: source-faithful narrative text views of the three reports (selected ATSB report narrative pages, and Cloudflare article prose excluding the HTML tables). Source-original tables are supplied separately under `source_tables/`.
- `D.text/Q1_A_aviation.txt`: one original ATSB report, contains E1, E2, E3.
- `D.text/Q1_B_software.txt`: the two original Cloudflare reports, contains E4, E5, E6.
- `D.text/Q1_C_combined.txt`: all three reports, contains E1-E6.
- `source_tables/`: full original event-timeline tables and Cloudflare June service impact table, cleaned only for line wrapping and CSV formatting.
- `TASKS.md`: mapping for Q1, Q2, and three four-example Q3 subsets.
- `manifest.json`: file names, original passage locations and source URLs.

## Validity notes

- `e.text` passages are **verbatim publisher prose**, with original PDF line breaks preserved. Cloudflare HTML paragraph text is extracted from saved HTML with normalized whitespace; no paraphrasing, rewriting, joining of nonconsecutive sentences, or synthetic event narration.
- For the original ATSB PDF, the supplied source text preserves line breaks from PDF extraction; the three passages must match their original source `.txt` exactly.
- Each `e.tab` is a **curated view** of rows in an original timeline plus, where necessary, investigative observations. The event timestamps, activity names, service impacts, and observation wording come from the uploaded originals. A derived row ID is added for auditability. These are reconstructed evidence tables, **not** raw flight-recorder exports, raw production logs, or independently verified causal proofs.
- E1-E3 are *different causal subchains of the same aviation accident*, not three independent flights. E5 and E6 arise from the same September software outage and use overlapping underlying timeline events, but describe different causal episodes. Q3 examples are therefore not statistically independent; the dataset tests structural generalization across domains, not independent prevalence.
- Source qualification (e.g., 'likely') is integral to the gold interpretation. The accident report cautions that findings are about contributing safety factors rather than assigning blame.
- Cloudflare's June report has richer service-impact records than incident timeline records; the E4 table therefore includes **original service-impact rows** in addition to timestamped events, with blank timestamps for those untimed impact records.
- This benchmark is curated for a small research evaluation. It should not be interpreted as a causal inference dataset in which arrows can be established from timestamps alone.

## Reuse and attribution

Original authors' wording and tables are credited in the `manifest.json` and individual source prose files. The ATSB PDF explicitly licenses most authored material under CC BY 3.0 Australia with attribution; check source exceptions for third-party materials. The Cloudflare sources are reproduced from the copies supplied by the user; external redistribution rights should be checked separately.

## Submission layout

`M.txt` and `T.txt` are the causal-chain definition used in the experiment.
`experiments/tasks/p3.jsonl` is the authoritative task mapping. Its example
text files contain a short ID header for the existing input loader. Other files
retain the source archive's layout. The local copies of third-party source
prose require rights review before public redistribution.
