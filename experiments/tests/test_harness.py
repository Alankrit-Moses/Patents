from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.agent import FrameworkQueryParser, setup_c_plan_preview
from experiments.agent import FrameworkAgent
from experiments.client import LLMResponse
from experiments.config import HarnessConfig
from experiments.manifest import build_manifest, build_tasks, manifest_counts
from experiments.prompts import generator_input_refs, materialize_generator_inputs
from experiments.reports import chunk_report
from experiments.reports import recover_exact_span
from experiments.manifest import extract_saved_excerpt, extract_saved_excerpts
from experiments.io_utils import read_text
from experiments.judge import _evaluate_exp1, _evaluate_exp2, _evaluate_exp3
from experiments.prompts import build_prompt
from experiments.robustness import summarize_robustness
from experiments.io_utils import write_jsonl


ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def complete(self, messages, **_kwargs):
        system = messages[0]["content"]
        if "DescriptionInductionAgent" in system:
            text = '{"T_hat":"a concrete before/after trajectory shift"}'
        elif "TextEvidenceAgent" in system:
            text = (
                '{"candidates":[{"excerpt":"Activity rose slowly before 2020 and declined after 2020.",'
                '"local_score":5,"matched_constraints":["explicit change point"]}]}'
            )
        elif "TextPatternVerifier" in system:
            text = '{"score":5,"matched_conditions":["before/after"],"failed_conditions":[],"rationale":"clear"}'
        else:
            text = "{}"
        return LLMResponse(text=text, raw={}, usage={})


class RecordingScoreClient:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.messages = []

    def complete(self, messages, **_kwargs):
        self.messages.append(messages)
        score = next(self.scores)
        return LLMResponse(text=f'{{"score":{score}}}', raw={}, usage={})


class SynthesisRecordingClient:
    def __init__(self):
        self.messages = []

    def complete(self, messages, **_kwargs):
        self.messages.append(messages)
        system = messages[0]["content"]
        if "PatternSynthesisAgent" in system:
            text = '{"M":"shared rule","T":"shared description"}'
        elif "DefinitionVerifier" in system:
            text = '{"score":5,"M_score":5,"T_score":5,"operationalizability":5,"matched_conditions":[],"failed_conditions":[],"rationale":""}'
        else:
            text = "{}"
        return LLMResponse(text=text, raw={}, usage={})


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = HarnessConfig(project_root=ROOT)
        cls.manifest = build_manifest(cls.config)
        cls.tasks = build_tasks(cls.manifest, cls.config)

    def test_current_manifest_counts(self):
        counts = manifest_counts(self.manifest, self.tasks)
        self.assertEqual(counts["patterns"], 2)
        self.assertEqual(counts["text_examples"], 12)
        self.assertEqual(counts["tabular_examples"], 12)
        self.assertEqual(counts["paired_examples"], 12)
        self.assertEqual(counts["tasks"], {"1": 6, "2": 12, "3": 6})

    def test_query_parser(self):
        parsed = FrameworkQueryParser().parse("D:(D.text,_)\nP:(M,_,{(?,_)[3]})")
        self.assertEqual(parsed["requested_count"], 3)
        self.assertIn("E.text", parsed["targets"])
        self.assertIn("M", parsed["known"])
        self.assertIn("D.tab", parsed["irrelevant"])
        exp2 = FrameworkQueryParser().parse("D:(D.text,_)\nP:(_,_,{(?,E.tab)})")
        self.assertIn("E.tab", exp2["known"])
        self.assertIn("E.text", exp2["targets"])
        exp3 = FrameworkQueryParser().parse(
            "D:(_,_)\nP:(?,?,{(E1.text,E1.tab),(E2.text,E2.tab),...})"
        )
        self.assertEqual(set(exp3["targets"]), {"M", "T"})
        self.assertIn("E.text", exp3["known"])
        self.assertIn("E.tab", exp3["known"])

    def test_experiment_two_materializer_has_no_explicit_gold_input(self):
        task = next(task for task in self.tasks if task["experiment"] == "2")
        inputs = materialize_generator_inputs(ROOT, task)
        refs = generator_input_refs(task)
        self.assertEqual(set(inputs), {"E.tab", "D.text"})
        self.assertEqual(set(refs), {"E.tab", "D.text"})
        self.assertNotIn("gold_e_text_path", refs)
        self.assertNotIn("E_tab_code", str(refs))
        self.assertNotIn(".xlsx", str(refs))

    def test_setup_a_is_minimal_prose_and_setup_b_is_tagged_framework_prompt(self):
        task = {
            "experiment": "1",
            "framework_query": "D:(D.text,_)\nP:(M,_,{(?,_)[3]})",
        }
        inputs = {"M": "PATTERN_SENTINEL", "D.text": "REPORT_SENTINEL"}
        prompt_a = build_prompt("A", task, inputs)[-1]["content"]
        self.assertNotIn("D:(D.text,_)", prompt_a)
        self.assertNotIn("<known_components>", prompt_a)
        self.assertNotIn("E.text", prompt_a)
        self.assertNotIn('"M"', prompt_a)
        self.assertNotIn('"T"', prompt_a)
        self.assertIn("Do not paraphrase or return the criterion itself", prompt_a)
        self.assertNotIn("boundary conditions", prompt_a.casefold())
        self.assertNotIn("operational", prompt_a.casefold())
        prompt_b = build_prompt("B", task, inputs)[-1]["content"]
        self.assertIn("<framework_guide>", prompt_b)
        self.assertIn("<framework_query>", prompt_b)
        self.assertIn("<known_components>", prompt_b)
        self.assertIn("<M>\nPATTERN_SENTINEL\n</M>", prompt_b)
        self.assertIn("<D_text>\nREPORT_SENTINEL\n</D_text>", prompt_b)
        self.assertIn("summarize the provided E.tab internally", prompt_b)
        self.assertIn("vague description", prompt_b)
        exp3_a = build_prompt(
            "A",
            {
                "experiment": "3",
                "framework_query": "D:(_,_)\nP:(?,?,{(E1.text,E1.tab),...})",
            },
            {
                "paired_examples": [
                    {
                        "example_id": "P1-E1",
                        "E.text": "Example passage.",
                        "E.tab": "year,value\n2020,1",
                    }
                ]
            },
        )[-1]["content"]
        self.assertNotIn('"M"', exp3_a)
        self.assertNotIn('"T"', exp3_a)
        self.assertIn('"mathematical_definition"', exp3_a)
        self.assertNotIn("excluding superficially", exp3_a)
        self.assertNotIn("accept the examples", exp3_a)
        exp2_a = build_prompt(
            "A",
            {"experiment": "2", "framework_query": "unused"},
            {"E.tab": "year,value\n2020,1", "D.text": "Example report passage."},
        )[-1]["content"]
        self.assertNotIn("entities, metric", exp2_a)
        self.assertNotIn("E.tab", exp2_a)
        self.assertIn("describes the same evidence as the table", exp2_a)

    def test_exp1_scores_each_requested_excerpt_and_hard_zeros_non_excerpts(self):
        with TemporaryDirectory(dir=ROOT) as temp_dir:
            base = Path(temp_dir)
            report_path = base / "report.txt"
            m_path = base / "M.txt"
            valid = "Activity rose before 2020 and declined after 2020."
            report_path.write_text(valid, encoding="utf-8")
            m_path.write_text("A before/after trajectory shift.", encoding="utf-8")
            task = {
                "report_path": report_path.relative_to(ROOT).as_posix(),
                "m_path": m_path.relative_to(ROOT).as_posix(),
                "target_count": 3,
            }
            result = {
                "parsed_output": {
                    "examples": [
                        {"excerpt": valid},
                        {"excerpt": "A before/after trajectory shift."},
                    ]
                }
            }
            client = RecordingScoreClient([4])
            metrics = _evaluate_exp1(client, self.config, task, result)
            self.assertEqual(metrics, {"E_text_alignment_scores": [4, 0, 0]})
        self.assertEqual(len(client.messages), 1)
        self.assertNotIn('{"score":1}', client.messages[0][-1]["content"])

    def test_exp2_uses_gold_pair_only_in_judge_and_returns_one_score(self):
        task = next(task for task in self.tasks if task["experiment"] == "2")
        inputs = materialize_generator_inputs(ROOT, task)
        gold = extract_saved_excerpt(ROOT / task["gold_e_text_path"])
        client = RecordingScoreClient([5])
        metrics = _evaluate_exp2(
            client,
            self.config,
            task,
            {"parsed_output": {"excerpt": gold}},
        )
        self.assertEqual(metrics, {"E_text_alignment_score": 5})
        judge_prompt = client.messages[0][-1]["content"]
        self.assertIn(inputs["E.tab"], judge_prompt)
        self.assertIn(gold, judge_prompt)
        self.assertNotIn("Comparison window used for E.tab", judge_prompt)
        self.assertIn("Do not return a rationale", judge_prompt)

    def test_exp3_scores_m_and_t_independently_without_heldouts(self):
        task = next(task for task in self.tasks if task["experiment"] == "3")
        client = RecordingScoreClient([4, 3])
        metrics = _evaluate_exp3(
            client,
            self.config,
            task,
            {"parsed_output": {"M": "generated math", "T": "generated text"}},
        )
        self.assertEqual(metrics, {"M_alignment_score": 4, "T_alignment_score": 3})
        self.assertEqual(len(client.messages), 2)
        prompts = "\n".join(call[-1]["content"] for call in client.messages)
        self.assertNotIn("Held-out", prompts)
        self.assertNotIn("Hard negative", prompts)

    def test_setup_c_is_action_compiled(self):
        plans = {task["experiment"]: setup_c_plan_preview(task) for task in self.tasks}
        self.assertIn("TextEvidenceAgent(map-reduce)", plans["1"])
        self.assertIn("TabSignalAgent", plans["2"])
        self.assertIn("PatternSynthesisAgent", plans["3"])

    def test_setup_c_executes_generic_text_plan_with_fake_client(self):
        config = HarnessConfig(project_root=ROOT, chunk_chars=1000, chunk_overlap_chars=50)
        task = {
            "task_id": "synthetic",
            "experiment": "1",
            "pattern_id": "P1",
            "framework_query": "D:(D.text,_)\nP:(M,_,{(?,_)[3]})",
        }
        inputs = {
            "M": "A trajectory differs before and after a concrete tau.",
            "D.text": "Activity rose slowly before 2020 and declined after 2020.",
        }
        result = FrameworkAgent(config, FakeClient()).run(task, inputs)
        self.assertEqual(
            result["parsed_output"]["examples"][0]["excerpt"], inputs["D.text"]
        )
        actions = [step["action"] for step in result["agent_trace"]]
        self.assertIn("TextEvidenceAgent", actions)
        self.assertIn("TextPatternVerifier", actions)
        self.assertIn("ReducerRanker", actions)

    def test_setup_c_definition_verifier_uses_only_visible_induction_pairs(self):
        client = SynthesisRecordingClient()
        task = {
            "task_id": "synthetic-exp3",
            "experiment": "3",
            "framework_query": "D:(_,_)\nP:(?,?,{(E1.text,E1.tab),...})",
            "heldout_examples": [
                {
                    "example_id": "LEAK_SENTINEL",
                    "e_text_path": "does/not/exist.txt",
                    "e_tab_path": "does/not/exist.csv",
                }
            ],
        }
        inputs = {
            "paired_examples": [
                {
                    "example_id": "VISIBLE_SENTINEL",
                    "E.text": "Visible report evidence.",
                    "E.tab": "year,value\n2020,1",
                }
            ]
        }
        result = FrameworkAgent(self.config, client).run(task, inputs)
        self.assertEqual(result["parsed_output"]["M"], "shared rule")
        all_prompts = "\n".join(
            message["content"] for call in client.messages for message in call
        )
        self.assertIn("VISIBLE_SENTINEL", all_prompts)
        self.assertNotIn("LEAK_SENTINEL", all_prompts)
        self.assertNotIn("Held-out positive", all_prompts)
        verifier = next(
            step for step in result["agent_trace"] if step["action"] == "DefinitionVerifier"
        )
        self.assertEqual(verifier["input_refs"], ["M", "T", "paired_examples"])

    def test_chunks_overlap(self):
        chunks = chunk_report("a" * 100, chunk_chars=40, overlap_chars=10)
        self.assertGreater(len(chunks), 1)
        self.assertLess(chunks[1].start_char, chunks[0].end_char)

    def test_saved_excerpts_are_recoverable_from_reports(self):
        reports = {
            name.casefold(): read_text(ROOT / path)
            for name, path in self.manifest.reports.items()
        }
        for pattern in self.manifest.patterns:
            for example in pattern.examples:
                if example.e_text_path:
                    excerpts = extract_saved_excerpts(ROOT / example.e_text_path)
                    self.assertGreaterEqual(len(excerpts), 1)
                    for excerpt in excerpts:
                        self.assertIsNotNone(
                            recover_exact_span(excerpt, reports[example.report_id.casefold()]),
                            example.example_id,
                        )

    def test_robustness_summary_reports_variance_best_of_k_and_retention(self):
        with TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "evaluations.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "task_id": "task-1",
                        "experiment": "2",
                        "setup": "A",
                        "pattern_id": "P1",
                        "sample_index": 0,
                        "metrics": {"E_text_alignment_score": 0},
                        "errors": [],
                    },
                    {
                        "task_id": "task-1",
                        "experiment": "2",
                        "setup": "A",
                        "pattern_id": "P1",
                        "sample_index": 1,
                        "metrics": {"E_text_alignment_score": 4},
                        "errors": [],
                    },
                ],
            )
            summary = summarize_robustness(path)
        metrics = summary["groups"][
            "experiment_2/setup_A/pattern_P1/E_text_alignment_score"
        ]
        self.assertEqual(metrics["mean"], 2.0)
        self.assertEqual(metrics["best_of_k_mean"], 4.0)
        self.assertEqual(metrics["success_at_k"], 1.0)
        self.assertEqual(metrics["exact_excerpt_retention_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
