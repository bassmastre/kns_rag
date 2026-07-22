from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


VERDICT_RE = re.compile(r"^\s*VERDICT\s*:\s*([OX])\s*$", re.IGNORECASE | re.MULTILINE)
REASON_RE = re.compile(r"^\s*REASON\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_judge_output(text: str) -> tuple[str, str]:
    verdict_match = VERDICT_RE.search(text or "")
    reason_match = REASON_RE.search(text or "")
    if not verdict_match or not reason_match:
        raise ValueError("judge output must contain 'VERDICT: O|X' and 'REASON: ...'")
    verdict = verdict_match.group(1).upper()
    reason = reason_match.group(1).strip()
    if not reason:
        raise ValueError("judge reason is empty")
    return verdict, reason


def summarize_judgments(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        o_count = sum(1 for row in rows if row.get("judge_verdict") == "O")
        x_count = sum(1 for row in rows if row.get("judge_verdict") == "X")
        error_count = sum(1 for row in rows if row.get("judge_verdict") not in {"O", "X"})
        graded = o_count + x_count
        return {
            "n": len(rows),
            "graded": graded,
            "O": o_count,
            "X": x_count,
            "errors": error_count,
            "accuracy": o_count / graded if graded else None,
        }

    result: dict[str, Any] = {"overall": summarize(records)}
    for output_key, record_key in (
        ("by_strategy", "strategy"),
        ("by_type", "qa_type"),
        ("by_section", "source_section"),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(record_key) or "unknown")].append(record)
        result[output_key] = {key: summarize(rows) for key, rows in sorted(groups.items())}
    return result
