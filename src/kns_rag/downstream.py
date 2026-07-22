from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

from .evaluation import normalize_match_text


ALLOWED_ERROR_TYPES = {
    "wrong_condition",
    "wrong_action",
    "wrong_completion_time",
    "connector_error",
    "missing_required_action",
    "unsupported_claim",
    "insufficient_context",
    "other",
}


ANSWER_SYSTEM_PROMPT = """You answer nuclear technical specification questions using only the supplied context.

Select only the provision whose LCO applicability and operating MODE match the question. Ignore provisions for other operating MODES, even when their wording or numerical limits are identical.

Preserve the exact logical relationships among Conditions, Required Actions, Completion Times, AND/OR connectors, notes, provisos, and alternatives. Do not combine a Required Action with a Completion Time from another row. Preserve all numerical values, units, inequalities, and logical connectors exactly; paraphrase only the surrounding explanation.

Answer in exactly this structure:

1. Applicable provision: State the applicable LCO number and operating MODE.
2. Answer: Give the direct answer concisely.
3. Conditions or alternatives: Include all provisos, required conditions, or alternatives necessary for the answer. Write “None” when none are relevant.

Keep each item concise.

If the supplied context does not contain a provision applicable to the operating MODE stated in the question, or is otherwise insufficient to answer reliably, respond with exactly:

INSUFFICIENT_CONTEXT
"""


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of answers about nuclear technical specifications.
Evaluate the candidate answer against the reference answer and the supplied retrieval context.
Ignore harmless wording differences. Penalize wrong Conditions, wrong Required Actions, wrong Completion Times, AND/OR errors, missing mandatory actions, and unsupported claims.
Return one JSON object only, with no markdown or surrounding prose."""


def format_contexts(contexts: list[dict[str, Any]]) -> str:
    if not contexts:
        return "[NO CONTEXT RETRIEVED]"
    return "\n\n".join(
        f"[{i}] {str(context.get('body') or '').strip()}"
        for i, context in enumerate(contexts, 1)
    )


def build_answer_messages(question: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question:\n{question.strip()}\n\nContext:\n{format_contexts(contexts)}",
        },
    ]


def build_judge_messages(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    contexts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    schema = {
        "verdict": "pass or fail",
        "correctness": "integer 0, 1, or 2",
        "completeness": "integer 0, 1, or 2",
        "relation_accuracy": "integer 0, 1, or 2",
        "unsupported_claim": "boolean",
        "error_types": ["zero or more allowed error labels"],
        "rationale": "brief explanation",
    }
    rubric = """Scoring rubric:
- correctness: 2 = all stated technical facts are correct; 1 = partly correct with a material factual error; 0 = fundamentally wrong.
- completeness: 2 = all answer elements required by the question/reference are present; 1 = partly complete; 0 = misses the central answer.
- relation_accuracy: 2 = Condition-Action, Action-Completion-Time, AND/OR, notes, and alternatives are all correctly bound; 1 = partly correct or relation not fully addressed; 0 = a material relationship is wrong.
- verdict is pass only when correctness=2, completeness=2, relation_accuracy=2, and unsupported_claim=false.
Allowed error_types: wrong_condition, wrong_action, wrong_completion_time, connector_error, missing_required_action, unsupported_claim, insufficient_context, other."""
    user_content = (
        f"{rubric}\n\nRequired JSON shape:\n{json.dumps(schema)}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Reference answer:\n{reference_answer.strip()}\n\n"
        f"Candidate answer:\n{candidate_answer.strip()}\n\n"
        f"Retrieval context:\n{format_contexts(contexts)}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first decodable JSON object from a model response."""
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("empty judge response")
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            value, _ = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge response did not contain a valid JSON object")


def _score(value: Any, field: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"judge field {field!r} must be an integer") from exc
    if score not in {0, 1, 2}:
        raise ValueError(f"judge field {field!r} must be 0, 1, or 2")
    return score


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    raise ValueError(f"judge field {field!r} must be a boolean")


def normalize_judgement(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one LLM-as-judge JSON result."""
    correctness = _score(value.get("correctness"), "correctness")
    completeness = _score(value.get("completeness"), "completeness")
    relation_accuracy = _score(value.get("relation_accuracy"), "relation_accuracy")
    unsupported_claim = _boolean(value.get("unsupported_claim", False), "unsupported_claim")

    raw_errors = value.get("error_types") or []
    if isinstance(raw_errors, str):
        raw_errors = [raw_errors]
    error_types = []
    for item in raw_errors:
        label = str(item).strip().lower()
        if not label:
            continue
        error_types.append(label if label in ALLOWED_ERROR_TYPES else "other")
    error_types = sorted(set(error_types))
    if unsupported_claim and "unsupported_claim" not in error_types:
        error_types.append("unsupported_claim")
        error_types.sort()

    expected_pass = (
        correctness == 2
        and completeness == 2
        and relation_accuracy == 2
        and not unsupported_claim
    )
    verdict = str(value.get("verdict") or "").strip().lower()
    if verdict not in {"pass", "fail"} or (verdict == "pass") != expected_pass:
        verdict = "pass" if expected_pass else "fail"

    return {
        "verdict": verdict,
        "passed": expected_pass,
        "correctness": correctness,
        "completeness": completeness,
        "relation_accuracy": relation_accuracy,
        "unsupported_claim": unsupported_claim,
        "error_types": error_types,
        "rationale": str(value.get("rationale") or "").strip(),
        "normalized_score": (correctness + completeness + relation_accuracy) / 6.0,
    }


def lexical_token_f1(reference: str, candidate: str) -> float:
    """Secondary lexical overlap diagnostic; not the main downstream score."""
    token_pattern = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*|<=|>=|<|>")
    reference_tokens = token_pattern.findall(normalize_match_text(reference))
    candidate_tokens = token_pattern.findall(normalize_match_text(candidate))
    if not reference_tokens or not candidate_tokens:
        return 0.0
    reference_counts = Counter(reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize_judgement_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate downstream judge rows by strategy, token budget, and QA type."""
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("strategy")), row.get("context_token_budget"))].append(row)

    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(group_rows)
        error_counts = Counter(
            error_type
            for row in group_rows
            for error_type in (row.get("error_types") or [])
        )
        return {
            "n": n,
            "pass_rate": sum(bool(row.get("passed")) for row in group_rows) / n,
            "mean_score": sum(float(row.get("normalized_score") or 0.0) for row in group_rows) / n,
            "correctness": sum(int(row.get("correctness") or 0) for row in group_rows) / (2 * n),
            "completeness": sum(int(row.get("completeness") or 0) for row in group_rows) / (2 * n),
            "relation_accuracy": sum(int(row.get("relation_accuracy") or 0) for row in group_rows) / (2 * n),
            "unsupported_claim_rate": sum(bool(row.get("unsupported_claim")) for row in group_rows) / n,
            "mean_lexical_f1": sum(float(row.get("lexical_f1") or 0.0) for row in group_rows) / n,
            "judge_parse_error_rate": sum(bool(row.get("judge_parse_error")) for row in group_rows) / n,
            "error_counts": dict(sorted(error_counts.items())),
        }

    summary: dict[str, Any] = {}
    for (strategy, budget), group_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1)
    ):
        strategy_summary = summary.setdefault(strategy, {})
        budget_key = "none" if budget is None else str(budget)
        values = summarize(group_rows)
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_type[str(row.get("qa_type") or "unknown")].append(row)
        values["by_type"] = {
            qa_type: summarize(type_rows)
            for qa_type, type_rows in sorted(by_type.items())
        }
        strategy_summary[budget_key] = values
    return summary
