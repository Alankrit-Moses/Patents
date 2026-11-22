from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.executor import AgenticExecutor, FrameworkQueryParser
from experiments.client import LLMResponse
from experiments.config import HarnessConfig
from experiments.task_construction import build_manifest, build_tasks, manifest_counts
from experiments.task_inputs import generator_input_refs, materialize_generator_inputs
from experiments.reports import chunk_report
from experiments.reports import recover_exact_span
from experiments.task_construction import extract_saved_excerpt, extract_saved_excerpts
from experiments.io_utils import read_text
from experiments.evaluation import _evaluate_exp1, _evaluate_exp2, _evaluate_exp3, _alignment_score
from unittest.mock import patch
from experiments.sampling import build_parser as build_sampling_parser
from experiments.sampling import summarize_samples
from experiments.sampling import _completed_key, _load_resume_state
from experiments.runner import prompt_version_for
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


class FakeAgentClient:
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

    def test_query_parser_preserves_variable_pair_slots(self):
        parsed = FrameworkQueryParser().parse(
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

    def test_default_judge_scale_accepts_five(self):
        client = RecordingScoreClient([5])
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_alignment_score(client, "Evaluate the answer."), 5)
        self.assertIn("integer from 1 to 5", client.messages[0][-1]["content"])

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
            client = RecordingScoreClient([3])
            metrics = _evaluate_exp1(client, self.config, task, result)
            self.assertEqual(metrics, {"E_text_alignment_scores": [3, 0, 0]})
        self.assertEqual(len(client.messages), 1)
        self.assertNotIn('{"score":1}', client.messages[0][-1]["content"])

    def test_exp2_judges_table_without_reference_and_returns_one_score(self):
        task = next(task for task in self.tasks if task["experiment"] == "2")
        inputs = materialize_generator_inputs(ROOT, task)
        gold = extract_saved_excerpt(ROOT / task["gold_e_text_path"])
        client = RecordingScoreClient([3])
        metrics = _evaluate_exp2(
            client,
            self.config,
            task,
            {"parsed_output": {"excerpt": gold}},
        )
        self.assertEqual(metrics, {"E_text_alignment_score": 3})
        judge_prompt = client.messages[0][-1]["content"]
        self.assertIn(inputs["E.tab"], judge_prompt)
        self.assertNotIn("Paired reference", judge_prompt)
        self.assertNotIn("Comparison window used for E.tab", judge_prompt)
        self.assertIn("Do not return a rationale", judge_prompt)

    def test_exp3_scores_m_and_t_independently_without_heldouts(self):
        task = next(task for task in self.tasks if task["experiment"] == "3")
        client = RecordingScoreClient([3, 2])
        metrics = _evaluate_exp3(
            client,
            self.config,
            task,
            {"parsed_output": {"M": "generated math", "T": "generated text"}},
        )
        self.assertEqual(metrics, {"M_alignment_score": 3, "T_alignment_score": 2})
        self.assertEqual(len(client.messages), 2)
        prompts = "\n".join(call[-1]["content"] for call in client.messages)
        for pair in materialize_generator_inputs(ROOT, task)["paired_examples"]:
            self.assertIn(pair["E.tab"], prompts)
            self.assertIn(pair["E.text"], prompts)
        self.assertNotIn("Canonical M:", prompts)
        self.assertNotIn("Held-out", prompts)
        self.assertNotIn("Hard negative", prompts)

    def test_submission_p3_tasks_load_and_q2_gold_is_grounded(self):
        import json
        path = ROOT / "experiments/tasks/p3.jsonl"
        tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({q: sum(t["experiment"] == q for t in tasks) for q in ("1", "2", "3")}, {"1": 3, "2": 6, "3": 3})
        for task in tasks:
            inputs = materialize_generator_inputs(ROOT, task)
            if task["experiment"] == "2":
                gold = extract_saved_excerpt(ROOT / task["gold_e_text_path"])
                self.assertIsNotNone(recover_exact_span(gold, inputs["D.text"]))
            elif task["experiment"] == "3":
                self.assertEqual(len(inputs["paired_examples"]), 4)
                self.assertTrue(all(pair["E.text"] and pair["E.tab"] for pair in inputs["paired_examples"]))

    def test_default_q3_m_uses_moderate_rubric(self):
        task = next(task for task in self.tasks if task["experiment"] == "3")
        client = RecordingScoreClient([4, 3])
        with patch.dict("os.environ", {}, clear=True):
            _evaluate_exp3(client, self.config, task, {"parsed_output": {"M": "math", "T": "text"}})
        m_prompt = client.messages[0][-1]["content"]
        self.assertIn("essential conditions are supported across the examples", m_prompt)
        self.assertIn("Return only", m_prompt)
        self.assertNotIn("Canonical M:", m_prompt)

    def test_agentic_executor_runs_framework_control_loop(self):
        config = HarnessConfig(project_root=ROOT, chunk_chars=1000, chunk_overlap_chars=50)
        task = {
            "task_id": "synthetic-agentic",
            "experiment": "1",
            "pattern_id": "P1",
            "framework_query": "D:(D.text,_)\nP:(M,_,{(?,_)[3]})",
            "target_count": 3,
        }
        inputs = {
            "M": "A trajectory differs before and after a concrete tau.",
            "D.text": "Activity rose slowly before 2020 and declined after 2020.",
        }
        result = AgenticExecutor(config, FakeAgentClient()).run(task, inputs)
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

    def test_sampling_summary_reports_variance_best_of_k_and_retention(self):
        with TemporaryDirectory(dir=ROOT) as temp_dir:
            path = Path(temp_dir) / "evaluations.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "task_id": "task-1",
                        "experiment": "2",
                        "setup": "C",
                        "pattern_id": "P1",
                        "sample_index": 0,
                        "metrics": {"E_text_alignment_score": 0},
                        "errors": [],
                    },
                    {
                        "task_id": "task-1",
                        "experiment": "2",
                        "setup": "C",
                        "pattern_id": "P1",
                        "sample_index": 1,
                        "metrics": {"E_text_alignment_score": 4},
                        "errors": [],
                    },
                ],
            )
            summary = summarize_samples(path)
        metrics = summary["groups"][
            "experiment_2/setup_C/pattern_P1/E_text_alignment_score"
        ]
        self.assertEqual(metrics["mean"], 2.0)
        self.assertEqual(metrics["best_of_k_mean"], 4.0)
        self.assertEqual(metrics["success_at_k"], 1.0)
        self.assertEqual(metrics["exact_excerpt_retention_rate"], 0.5)

    def test_sampling_defaults_to_three_total_attempts_per_sample(self):
        args = build_sampling_parser().parse_args(["run"])
        self.assertEqual(args.max_attempts, 3)

    def test_prompt_version_matches_released_executor(self):
        self.assertEqual(prompt_version_for(), "v2-agentic")

    def test_sampling_resume_accepts_historical_run_metadata(self):
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
            path = Path(temp_dir) / "outputs.jsonl"
            write_jsonl(path, records)
            _run_id, completed = _load_resume_state(path)
        self.assertIn(
            _completed_key("exp1-P1-genAI", "v2-agentic", 0), completed
        )
        self.assertIn(
            _completed_key("exp1-P1-genAI", "v2-agentic", 1), completed
        )


if __name__ == "__main__":
    unittest.main()
