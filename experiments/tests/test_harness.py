from __future__ import annotations

import unittest
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


if __name__ == "__main__":
    unittest.main()
