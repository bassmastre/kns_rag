from __future__ import annotations

import unittest

from kns_rag.downstream import (
    extract_json_object,
    lexical_token_f1,
    normalize_judgement,
    summarize_judgement_rows,
)


class DownstreamJudgingTests(unittest.TestCase):
    def test_extracts_json_from_surrounding_text(self) -> None:
        parsed = extract_json_object(
            '```json\n{"verdict":"pass","correctness":2,"completeness":2,'
            '"relation_accuracy":2,"unsupported_claim":false,"error_types":[]}\n```'
        )
        self.assertEqual(parsed["verdict"], "pass")

    def test_normalizes_string_false_and_enforces_pass_rule(self) -> None:
        judgement = normalize_judgement(
            {
                "verdict": "fail",
                "correctness": 2,
                "completeness": 2,
                "relation_accuracy": 2,
                "unsupported_claim": "false",
                "error_types": [],
                "rationale": "Equivalent answer.",
            }
        )
        self.assertTrue(judgement["passed"])
        self.assertEqual(judgement["verdict"], "pass")

    def test_unknown_error_label_is_mapped_to_other(self) -> None:
        judgement = normalize_judgement(
            {
                "verdict": "fail",
                "correctness": 1,
                "completeness": 2,
                "relation_accuracy": 2,
                "unsupported_claim": False,
                "error_types": ["invented_label"],
            }
        )
        self.assertEqual(judgement["error_types"], ["other"])

    def test_lexical_f1_is_bounded(self) -> None:
        score = lexical_token_f1(
            "Restore the loop within 72 hours.",
            "Restore the required loop within 72 hours.",
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_summary_groups_strategy_budget_and_type(self) -> None:
        rows = [
            {
                "strategy": "condition_aware",
                "context_token_budget": 512,
                "qa_type": "completion_time",
                "passed": True,
                "normalized_score": 1.0,
                "correctness": 2,
                "completeness": 2,
                "relation_accuracy": 2,
                "unsupported_claim": False,
                "lexical_f1": 0.8,
                "judge_parse_error": False,
                "error_types": [],
            },
            {
                "strategy": "condition_aware",
                "context_token_budget": 512,
                "qa_type": "completion_time",
                "passed": False,
                "normalized_score": 0.5,
                "correctness": 1,
                "completeness": 1,
                "relation_accuracy": 1,
                "unsupported_claim": False,
                "lexical_f1": 0.4,
                "judge_parse_error": False,
                "error_types": ["wrong_completion_time"],
            },
        ]
        values = summarize_judgement_rows(rows)["condition_aware"]["512"]
        self.assertEqual(values["n"], 2)
        self.assertEqual(values["pass_rate"], 0.5)
        self.assertEqual(values["error_counts"], {"wrong_completion_time": 1})


if __name__ == "__main__":
    unittest.main()
