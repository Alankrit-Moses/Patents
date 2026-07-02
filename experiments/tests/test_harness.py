from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.agent_v2 import BlackboardV2, FrameworkAgentV2, FrameworkQueryParserV2
from experiments.agent_v2_full import FrameworkAgentV2FullContext
from experiments.client import LLMResponse
from experiments.config import HarnessConfig
from experiments.manifest import build_manifest, build_tasks, manifest_counts
from experiments.prompts import generator_input_refs, materialize_generator_inputs
from experiments.reports import chunk_report
from experiments.reports import recover_exact_span
from experiments.manifest import extract_saved_excerpt, extract_saved_excerpts
from experiments.io_utils import read_text
from experiments.judge import _evaluate_exp1, _evaluate_exp2, _evaluate_exp3
from experiments.robustness import build_parser as build_robustness_parser
from experiments.robustness import summarize_robustness
from experiments.robustness import _completed_key, _load_resume_state
from experiments.runner_v2 import prompt_version_for
from experiments.io_utils import write_jsonl


ROOT = Path(__file__).resolve().parents[2]


class RecordingScoreClient:
    def __init__(self, scores):
        self.scores = iter(scores)
        self.messages = []

    def complete(self, messages, **_kwargs):
        self.messages.append(messages)
        score = next(self.scores)
        return LLMResponse(text=f'{{"score":{score}}}', raw={}, usage={})


class AgenticV2Client:
    def __init__(self):
        self.framework_calls = 0
        self.messages = []

    def complete(self, messages, **_kwargs):
        self.messages.append(messages)
        system = messages[0]["content"]
        if "Framework Agent" in system:
            self.framework_calls += 1
            if self.framework_calls == 1:
                text = (
                    '{"action":"invoke_agent","agent":"E.text-Finder","reason":"find exact spans",'
                    '"task_packet":{"goal":"locate E.text","query_context":{'
                    '"framework_query":"D:(D.text,_) P:(M,_,{(?,_)[3]})",'
                    '"active_pattern_ids":["P"],'
                    '"variable_slots":["P.E[1].E.text","P.E[2].E.text","P.E[3].E.text"],'
                    '"concrete_slots":["P.M","D.text"],"not_provided_slots":["D.tab","P.T","P.E[1].E.tab"]},'
                    '"materials_needed":["D.text","M"],"working_memory_inputs":[],"constraints":[]}}'
                )
            else:
                text = (
                    '{"action":"return_final","final_assignments":{'
                    '"P.E[1].E.text":"Activity rose slowly before 2020 and declined after 2020.",'
                    '"P.E[2].E.text":"Activity rose slowly before 2020 and declined after 2020.",'
                    '"P.E[3].E.text":"Activity rose slowly before 2020 and declined after 2020."},'
                    '"reason":"all ? slots have candidate assignments"}'
                )
        elif "E.text-Finder agent" in system:
            text = (
                '{"agent":"E.text-Finder","status":"success","candidates":{'
                '"P.E[1].E.text":[{"excerpt":"Activity rose slowly before 2020 and declined after 2020.",'
                '"confidence":0.9,"match_basis":["pattern_behavior"],"notes":""}],'
                '"P.E[2].E.text":[{"excerpt":"Activity rose slowly before 2020 and declined after 2020.",'
                '"confidence":0.9,"match_basis":["pattern_behavior"],"notes":""}],'
                '"P.E[3].E.text":[{"excerpt":"Activity rose slowly before 2020 and declined after 2020.",'
                '"confidence":0.9,"match_basis":["pattern_behavior"],"notes":""}]}}'
            )
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

    def test_query_parser_v2_preserves_variable_pair_slots(self):
        parsed = FrameworkQueryParserV2().parse(
            "D:(D.text,_)\nP1:(_,_,{(?,E.tab_1),(?,E.tab_2),(?,E.tab_3)})"
        )
        self.assertEqual(
            parsed.variable_slots,
            ["P1.E[1].E.text", "P1.E[2].E.text", "P1.E[3].E.text"],
        )
        self.assertIn("P1.E[1].E.tab", parsed.concrete_slots)
        self.assertIn("P1.M", parsed.not_provided_slots)

    def test_experiment_two_materializer_has_no_explicit_gold_input(self):
        task = next(task for task in self.tasks if task["experiment"] == "2")
        inputs = materialize_generator_inputs(ROOT, task)
        refs = generator_input_refs(task)
        self.assertEqual(set(inputs), {"E.tab", "D.text"})
        self.assertEqual(set(refs), {"E.tab", "D.text"})
        self.assertNotIn("gold_e_text_path", refs)
        self.assertNotIn("E_tab_code", str(refs))
        self.assertNotIn(".xlsx", str(refs))

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

    def test_setup_c_v2_executes_framework_agent_control_loop(self):
        config = HarnessConfig(project_root=ROOT, chunk_chars=1000, chunk_overlap_chars=50)
        task = {
            "task_id": "synthetic-v2",
            "experiment": "1",
            "pattern_id": "P1",
            "framework_query": "D:(D.text,_)\nP:(M,_,{(?,_)[3]})",
            "target_count": 3,
        }
        inputs = {
            "M": "A trajectory differs before and after a concrete tau.",
            "D.text": "Activity rose slowly before 2020 and declined after 2020.",
        }
        result = FrameworkAgentV2(config, AgenticV2Client()).run(task, inputs)
        self.assertEqual(len(result["parsed_output"]["examples"]), 3)
        actions = [step["action"] for step in result["agent_trace"]]
        self.assertIn("FrameworkAgent", actions)
        self.assertIn("E.text-Finder", actions)
        # The Verifier has been removed: specialist outputs are used directly.
        self.assertNotIn("Verifier", actions)
        self.assertEqual(
            result["parsed_output"]["examples"][0]["excerpt"],
            inputs["D.text"],
        )

    def test_setup_c_v2_full_context_finder_reads_whole_report_in_one_call(self):
        # chunk_chars is small enough that the map-reduce variant would split
        # this report; the full-context variant must still make exactly one
        # finder call containing the entire report.
        config = HarnessConfig(project_root=ROOT, chunk_chars=100, chunk_overlap_chars=10)
        span = "Filings rose slowly before 2019 and accelerated sharply after 2019."
        report = ("Background paragraph about unrelated matters. " * 5) + span
        self.assertGreater(len(report), config.chunk_chars)
        task = {
            "task_id": "synthetic-v2-full",
            "experiment": "2",
            "pattern_id": "P1",
            "framework_query": "D:(D.text,_)\nP:(_,_,{(?,E.tab)})",
            "target_count": 1,
        }
        inputs = {
            "E.tab": "entity,metric,2018,2020\nfilings,count,10,90",
            "D.text": report,
        }

        class FullContextClient:
            def __init__(self):
                self.finder_calls = 0
                self.finder_messages = []
                self.framework_calls = 0

            def complete(self, messages, **_kwargs):
                system = messages[0]["content"]
                if "Framework Agent" in system:
                    self.framework_calls += 1
                    if self.framework_calls == 1:
                        text = (
                            '{"action":"invoke_agent","agent":"E.text-Finder","reason":"find span",'
                            '"task_packet":{"goal":"locate E.text","query_context":{'
                            '"framework_query":"D:(D.text,_) P:(_,_,{(?,E.tab)})",'
                            '"active_pattern_ids":["P"],'
                            '"variable_slots":["P.E[1].E.text"],'
                            '"concrete_slots":["D.text","P.E[1].E.tab"],'
                            '"not_provided_slots":["D.tab","P.M","P.T"]},'
                            '"materials_needed":["D.text","P.E[1].E.tab"],'
                            '"working_memory_inputs":[],"constraints":[]}}'
                        )
                    else:
                        text = (
                            '{"action":"return_final","final_assignments":{'
                            f'"P.E[1].E.text":"{span}"}},'
                            '"reason":"the ? slot has a candidate assignment"}'
                        )
                elif "ENTIRE report" in system:
                    self.finder_calls += 1
                    self.finder_messages.append(messages)
                    text = (
                        '{"agent":"E.text-Finder","status":"success","selected":['
                        f'{{"excerpt":"{span}","confidence":0.9,'
                        '"match_basis":["trend"],"notes":""}]}'
                    )
                else:
                    text = "{}"
                return LLMResponse(text=text, raw={}, usage={})

        client = FullContextClient()
        result = FrameworkAgentV2FullContext(config, client).run(task, inputs)
        self.assertEqual(client.finder_calls, 1)
        self.assertIn(report, client.finder_messages[0][-1]["content"])
        self.assertEqual(result["parsed_output"], {"excerpt": span})
        self.assertEqual(result["selected_chunks"], ["full-report"])

    def test_setup_c_v2_full_context_finder_errors_instead_of_truncating(self):
        config = HarnessConfig(
            project_root=ROOT,
            chunk_chars=100,
            chunk_overlap_chars=10,
            context_limit_tokens=64,
        )
        task = {
            "task_id": "synthetic-v2-full-overflow",
            "experiment": "2",
            "pattern_id": "P1",
            "framework_query": "D:(D.text,_)\nP:(_,_,{(?,E.tab)})",
            "target_count": 1,
        }
        inputs = {"E.tab": "a,b\n1,2", "D.text": "words " * 500}
        agent = FrameworkAgentV2FullContext(config, object())
        agent.etext_slots = ["P.E[1].E.text"]
        with self.assertRaises(ValueError):
            agent._invoke_e_text_finder(
                {"materials_needed": ["D.text"], "query_context": {}},
                inputs,
                BlackboardV2(),
            )

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

    def test_robustness_defaults_to_three_total_attempts_per_sample(self):
        args = build_robustness_parser().parse_args(["run"])
        self.assertEqual(args.max_attempts, 3)

    def test_prompt_version_distinguishes_every_setup_variant(self):
        versions = {
            prompt_version_for("A", "query-only", "map-reduce"),
            prompt_version_for("B", "query-only", "map-reduce"),
            prompt_version_for("B", "resolved", "map-reduce"),
            prompt_version_for("C", "query-only", "map-reduce"),
            prompt_version_for("C", "query-only", "full-context"),
        }
        # All five runnable configurations must map to distinct tags so resume
        # never conflates them in a shared output file.
        self.assertEqual(len(versions), 5)
        self.assertEqual(prompt_version_for("C", "query-only", "map-reduce"), "v2-agentic")
        self.assertEqual(prompt_version_for("C", "query-only", "full-context"), "v2-agentic-full")

    def test_robustness_resume_keeps_c_variants_distinct(self):
        # A file containing completed map-reduce C samples must NOT cause the
        # full-context C samples to be skipped on resume, and vice versa.
        records = [
            {
                "task_id": "exp1-P1-genAI",
                "prompt_version": "v2-agentic",
                "sample_index": 0,
                "robustness_run_id": "run-xyz",
                "errors": [],
            },
            {
                "task_id": "exp1-P1-genAI",
                "prompt_version": "v2-agentic",
                "sample_index": 1,
                "robustness_run_id": "run-xyz",
                "errors": [],
            },
        ]
        with TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            write_jsonl(path, records)
            _run_id, completed = _load_resume_state(path)
        # map-reduce sample already present -> completed
        self.assertIn(
            _completed_key("exp1-P1-genAI", "v2-agentic", 0), completed
        )
        # full-context sample for the same task/index -> NOT completed
        self.assertNotIn(
            _completed_key("exp1-P1-genAI", "v2-agentic-full", 0), completed
        )


if __name__ == "__main__":
    unittest.main()
