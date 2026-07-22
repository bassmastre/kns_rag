from __future__ import annotations

import unittest

from kns_rag.judging import parse_judge_output, summarize_judgments
from kns_rag.prompts import make_judge_prompt, make_llm_only_prompt, make_rag_prompt


class PromptTests(unittest.TestCase):
    def test_rag_prompt_requires_section_and_complete_logic(self) -> None:
        prompt = make_rag_prompt("Question?", [{"body": "LCO 3.4.5 text"}])
        self.assertIn("State the applicable LCO section number explicitly", prompt)
        self.assertIn("every required AND-bound", prompt)
        self.assertIn("LCO 3.4.5 text", prompt)

    def test_llm_only_prompt_has_no_context(self) -> None:
        prompt = make_llm_only_prompt("Question?")
        self.assertIn("No retrieved context is provided", prompt)
        self.assertNotIn("Context:\n", prompt)

    def test_judge_prompt_interpolates_strict_fields(self) -> None:
        prompt = make_judge_prompt(
            question="Q",
            reference_answer="R",
            evidence_keywords=[["a", "b"]],
            source_section="3.4.5",
            generated_answer="A",
        )
        self.assertIn("CORRECT SOURCE SECTION: 3.4.5", prompt)
        self.assertIn('[\"a\", \"b\"]', prompt)
        self.assertIn("VERDICT: O or X", prompt)


class JudgeTests(unittest.TestCase):
    def test_parse_judge_output(self) -> None:
        verdict, reason = parse_judge_output("VERDICT: X\nREASON: Rule 3 triggered.")
        self.assertEqual(verdict, "X")
        self.assertEqual(reason, "Rule 3 triggered.")

    def test_parse_rejects_unstructured_output(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output("Probably correct")

    def test_summary_uses_only_graded_rows_for_accuracy(self) -> None:
        summary = summarize_judgments(
            [
                {"strategy": "a", "qa_type": "x", "source_section": "3.4.5", "judge_verdict": "O"},
                {"strategy": "a", "qa_type": "x", "source_section": "3.4.5", "judge_verdict": "X"},
                {"strategy": "a", "qa_type": "x", "source_section": "3.4.5", "judge_verdict": None},
            ]
        )
        self.assertEqual(summary["overall"]["graded"], 2)
        self.assertEqual(summary["overall"]["errors"], 1)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
