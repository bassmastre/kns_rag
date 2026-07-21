from __future__ import annotations

import unittest

from kns_rag.evaluation import token_budget_stats
from kns_rag.retrieval import select_ranked_prefix_by_token_budget


class TokenBudgetSelectionTests(unittest.TestCase):
    def test_selects_highest_ranked_prefix_that_fits(self) -> None:
        ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]
        selected = select_ranked_prefix_by_token_budget(
            ranked,
            token_counts=[100, 200, 50],
            max_token_budget=300,
        )
        self.assertEqual(
            selected,
            [(0, 0.9, 100, 100), (1, 0.8, 200, 300)],
        )

    def test_does_not_skip_oversized_higher_ranked_chunk(self) -> None:
        ranked = [(0, 0.9), (1, 0.8), (2, 0.7)]
        selected = select_ranked_prefix_by_token_budget(
            ranked,
            token_counts=[100, 250, 50],
            max_token_budget=300,
        )
        self.assertEqual(selected, [(0, 0.9, 100, 100)])

    def test_budget_metrics_use_cumulative_tokens(self) -> None:
        results = [
            {"rank": 1, "body": "alpha", "cumulative_tokens": 100},
            {"rank": 2, "body": "beta", "cumulative_tokens": 300},
        ]
        stats = token_budget_stats(results, [["alpha"], ["beta"]], [100, 300])
        self.assertTrue(stats["hit@100t"])
        self.assertEqual(stats["recall@100t"], 0.5)
        self.assertFalse(stats["set_recall@100t"])
        self.assertTrue(stats["set_recall@300t"])


if __name__ == "__main__":
    unittest.main()
