Input/output inspection export
==============================

Folder structure
----------------

This folder is organized task-first, then by run condition, then by setup:

input_output/
  <task_id>/
    <condition>/
      setup_A/
        input.txt
        output.txt
        evaluator_prompt.txt
        evaluation.json
      setup_B/
        ...
      setup_C/
        ...

For stochastic runs with multiple samples, each setup contains sample folders:

input_output/
  <task_id>/
    temperature_zero_point_two/
      setup_A/
        sample_1/
          input.txt
          output.txt
          evaluator_prompt.txt
          evaluation.json

For example, the Setup A files for a deterministic temperature-zero run of task exp1-P1-genAI are stored as:

input_output/exp1-P1-genAI/temperature_zero/setup_A/

Files inside each setup folder
------------------------------

- input.txt: prompt sent to the setup, or Setup C's typed orchestrator input packet.
- output.txt: full generator output that was evaluated by the judge.
- evaluator_prompt.txt: judge prompt(s), or exact-membership gate note when no LLM judge call was made.
- evaluation.json: raw judge evaluation record with score fields.

Setups
------

- setup_A: minimal-prose single-shot baseline. It receives the task evidence in ordinary prose and returns JSON.
- setup_B: framework-aware single-shot prompt. It receives the same task-visible evidence, but with the framework primer, typed query, tagged components, and structured output constraints.
- setup_C: framework-compiled workflow. It receives the typed task packet, uses the common orchestrator and reusable worker actions, and returns only the requested target component(s). For Setup C, input.txt records the orchestrator input packet rather than a single LLM prompt.

Experiments
-----------

- Experiment 1, M -> E.text[3]: given a mathematical pattern definition and one report, return three exact report excerpts instantiating the pattern.
- Experiment 2, E.tab -> E.text: given one compact tabular example and the corresponding report, return the exact report excerpt describing the same evidence.
- Experiment 3, (E.text,E.tab)^4 -> M,T: given four paired text/table examples from a pattern, induce the mathematical definition M and natural-language description T.

Patterns
--------

- P1: trajectory-shift pattern, where an entity/metric changes behavior around an identifiable time or event.
- P2: comparative-growth pattern, where comparable entities are compared by CAGR over the same interval.

Conditions
----------

Condition folders separate different run settings. The deterministic run is stored under `temperature_zero`. Robustness samples at temperature 0.2 are stored under `temperature_zero_point_two`, with one `sample_<n>` folder per sampled output.

The most recently exported condition for this folder is `temperature_zero_point_two`.

Evaluation
----------

Text-output experiments first apply an exact-membership gate: if an expected report excerpt is not recoverable verbatim from the report, the score is 0 and evaluator_prompt.txt records that no LLM judge call was made. Otherwise, the judge prompt asks for a 1-5 alignment score. Experiment 3 separately judges generated M and T against the canonical pattern definitions.

{
  "results_path": "C:\\Users\\Alankrit Moses\\OneDrive\\Desktop\\AI\\WIPO\\experiments\\artifacts\\robustness\\results-mistral-small-3.2-24b-temp0.2-k3-20260624-140907.jsonl",
  "evaluations_path": "C:\\Users\\Alankrit Moses\\OneDrive\\Desktop\\AI\\WIPO\\experiments\\artifacts\\evaluations\\evaluations-mistral-small-3.2-24b-temp0.2-k3-20260624.jsonl",
  "tasks_path": "C:\\Users\\Alankrit Moses\\OneDrive\\Desktop\\AI\\WIPO\\experiments\\artifacts\\tasks.jsonl",
  "output_dir": "C:\\Users\\Alankrit Moses\\OneDrive\\Desktop\\AI\\WIPO\\experiments\\artifacts\\input_output",
  "condition": "temperature_zero_point_two",
  "records_exported": 216,
  "setups": [
    "A",
    "B",
    "C"
  ],
  "missing_task_records": [],
  "missing_evaluations": []
}
