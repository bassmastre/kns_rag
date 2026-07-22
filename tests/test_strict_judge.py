from __future__ import annotations

import unittest

from kns_rag.strict_judge import (
    build_llm_only_messages,
    make_judge_prompt,
    parse_judge_output,
    summarize_judgements,
)


class StrictJudgeTests(unittest.TestCase):
    def test_parse_o(self) -> None:
        verdict, reason = parse_judge_output(
            "VERDICT: O\nREASON: All four O conditions are satisfied."
        )
        self.assertEqual(verdict, "O")
        self.assertIn("satisfied", reason)

    def test_parse_x(self) -> None:
        verdict, reason = parse_judge_output(
            "VERDICT: X\nREASON: Rule 3 triggered because one required action is missing."
        )
        self.assertEqual(verdict, "X")
        self.assertIn("Rule 3", reason)

    def test_rejects_malformed_output(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output("O")

    def test_prompt_contains_required_fields(self) -> None:
        prompt = make_judge_prompt(
            question="What applies?",
            reference_answer="Condition B applies.",
            evidence_keywords=[["Condition B", "Immediately"]],
            source_section="3.4.6",
            generated_answer="LCO 3.4.6, Condition B applies immediately.",
        )
        self.assertIn("CORRECT SOURCE SECTION: 3.4.6", prompt)
        self.assertIn('[["Condition B", "Immediately"]]', prompt)
        self.assertIn("VERDICT: O or X", prompt)

    def test_llm_only_has_no_context(self) -> None:
        messages = build_llm_only_messages("What applies?")
        self.assertEqual(len(messages), 2)
        self.assertNotIn("Context:", messages[1]["content"])

    def test_summary(self) -> None:
        summary = summarize_judgements(
            [
                {
                    "strategy": "llm_only",
                    "context_token_budget": 0,
                    "qa_type": "definition",
                    "source_section": "3.4.5",
                    "judge_verdict": "O",
                    "judge_error": None,
                },
                {
                    "strategy": "llm_only",
                    "context_token_budget": 0,
                    "qa_type": "definition",
                    "source_section": "3.4.5",
                    "judge_verdict": "X",
                    "judge_error": None,
                },
            ]
        )
        self.assertEqual(summary["overall"]["n"], 2)
        self.assertEqual(summary["overall"]["O"], 1)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
