# Agentic executor protocol

The executor is an agentic loop driven by an orchestrator and a fixed catalog
of component agents. It uses the framework query to decide which component to
produce next and exchanges state through a persistent blackboard.

## Framework components

```text
D = (D.text, D.tab)
P = (M, T, E)
E = {(e.text_i, e.tab_i)}
```

- `M` is a mathematical pattern definition.
- `T` is a textual pattern description.
- `E` is a set of paired textual and tabular examples.
- `_` marks a slot that is neither supplied nor requested.
- `?` marks a target slot.
- `[n]` requests a fixed number of values.

## Orchestrator

The orchestrator receives the framework specification, typed query, parsed
slot states, material inventory, current blackboard, and component-agent
catalog. It does not receive complete reports or tables by default. It either
produces a task packet for one agent or returns assignments for the target
slots.

Every task packet uses the same fields: `goal`, `query_context`,
`materials_needed`, `working_memory_inputs`, and `constraints`. Material names
are canonical framework slots; the runtime resolves them before constructing
the selected agent's prompt.

## Component agents

- **M-Inducer** produces a requested `M` or an intermediate `M_hat`.
- **T-Inducer** produces a requested `T` or an intermediate `T_hat`.
- **e.text-finder** searches overlapping report chunks in parallel and pools
  candidate passages.
- **e.text-selector** ranks that pool by candidate identifier; the runtime then
  writes the selected source passages to the target `e.text` slots.

The selector is internal to the e.text retrieval procedure. There is no
verifier and no e.tab-specialist in the evaluated executor.

The exact evaluated prompts use `Framework Agent` for the orchestrator and
`E.text-Finder`/`E.text Selector` for the two retrieval roles; the alternate
spellings above follow the paper's terminology.

## Blackboard and runtime loop

The blackboard stores intermediate `M_hat` and `T_hat` components, candidate
target assignments, resolved target slots, and the execution trace. The
runtime repeatedly asks the orchestrator for the next task packet, executes the
selected component agent, updates the blackboard, and returns only the slots
marked `?` once all targets have assignments.

The exact prompts and JSON output contracts are defined in `executor.py`; judge
prompts are defined in `evaluation.py`.
