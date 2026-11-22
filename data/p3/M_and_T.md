# P3 — Documented Causal / Dependency Propagation Chain

## M — Mathematical definition

Let V_c denote the recorded events, states, or conditions associated with a particular case c. Let Causal_c(a,b) mean that the source investigation explicitly supports a directed influence of a on b, such as causing, enabling, aggravating, or propagating a disruption. Let t(v) be an observed time when one is available; conditions without timestamps need not have a t(v).

An instance of P3 is an ordered sequence of k >= 3 distinct events or conditions:

  (v_1, v_2, ..., v_k),   k >= 3,

such that for every i from 1 to k−1, Causal_c(v_i, v_{i+1}) holds. When both times are available and comparable, t(v_i) <= t(v_{i+1}), subject to the source's chronology. Evidence can include original event logs, recorded observations, configuration data, and the source's causal analysis.

Temporal succession alone is never sufficient. The original source must support each dependency link, and qualifiers ("likely," "contributed to," etc.) must be preserved. Different chains may share events and may belong to one incident.

## T — Textual description

A multi-stage failure or consequence chain in which an initiating fault, action, or condition produces an intermediate failure or operational change that causes or contributes to a later outcome. The pattern is about a documented causal or functional dependency, not merely events happening one after another. It can describe aircraft operations or failures spreading through software systems.

## Evidence conventions

`e.text` is a contiguous original report passage (only HTML whitespace interpretation or PDF hard line breaks differ from page rendering). `e.tab` is a selection of actual timestamped events and original investigative/service-impact observations; no synthetic cases, inferred timestamps, or precomputed pattern arrows are included. Some causal edges are expressed only in the report prose, not proved from the timestamps.
