from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


LLM_ONLY_SYSTEM_PROMPT = """You answer nuclear plant Technical Specifications (LCO) questions from your own model knowledge, without retrieved context.

Give a direct and complete answer. Explicitly state the applicable LCO section number. Include every mandatory AND-bound action or condition, every OR alternative requested by the question, the correct Condition letter and logical structure, and all numeric values, inequalities, units, and Completion Times needed for the answer.

Do not invent unsupported details. If you cannot answer reliably, respond with exactly:

INSUFFICIENT_CONTEXT
"""


JUDGE_PROMPT_TEMPLATE = """You are grading whether a generated answer to a nuclear plant Technical
Specifications (LCO) question is correct, using the reference answer as
ground truth.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference_answer}

EVIDENCE KEYWORDS (nested list; inner list = AND group, groups joined by OR):
{evidence_keywords}

CORRECT SOURCE SECTION: {source_section}

GENERATED ANSWER TO GRADE:
{generated_answer}

Grade the GENERATED ANSWER as O (correct) or X (incorrect) using these rules:

Grade O only if ALL of the following hold:
1. The LCO section cited in the generated answer includes {source_section}
   (citing additional sections alongside it is fine).
2. Every AND-bound requirement in the reference answer is present.
3. Every OR-alternative the question asks for is present.
4. All numeric values, conditions, and actions match the reference answer
   without contradiction.

Grade X if ANY of the following hold:
1. The answer is "INSUFFICIENT_CONTEXT", a refusal, or otherwise does not
   commit to an answer.
2. The generated answer cites a wrong or nonexistent LCO section as its
   SOLE source, even if the resulting value happens to match the reference
   answer (coincidental match from an unrelated section is not credit-
   worthy — this includes cases where the same phrase appears in a
   different LCO's parallel provision).
3. Any required AND-bound item is missing.
4. Any required OR-alternative is missing.
5. The answer confuses or misstates the Condition letter / logical
   structure (e.g., attributes the requirement to the wrong Condition).
6. The answer is garbled, incoherent, or internally contradictory.
7. The answer states a value or condition not supported by the evidence
   keywords or reference answer.

Do NOT penalize:
- Mentioning additional (correct or incorrect) sections alongside the
  correct one, as long as the correct section is explicitly and
  unambiguously included.
- Paraphrasing, informal section references (e.g., "LCO E" instead of
  "LCO 3.4.12, Condition E"), or stylistic differences, as long as the
  content is accurate.

Output strictly in this format:
VERDICT: O or X
REASON: <one sentence, citing which rule triggered the verdict>
"""


_VERDICT_LINE = re.compile(r"^VERDICT:\s*([OX])\s*$", re.IGNORECASE)
_REASON_LINE = re.compile(r"^REASON:\s*(.+?)\s*$", re.IGNORECASE)


def build_llm_only_messages(question: str) -> list[dict[str, str]]:
    """Build a closed-book generation request with no retrieved context."""
    return [
        {"role": "system", "content": LLM_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question.strip()}"},
    ]


def make_judge_prompt(
    *,
    question: str,
    reference_answer: str,
    evidence_keywords: list[list[str]],
    source_section: str,
    generated_answer: str,
) -> str:
    """Render the experiment's strict binary judge prompt."""
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        reference_answer=reference_answer.strip(),
        evidence_keywords=json.dumps(evidence_keywords, ensure_ascii=False),
        source_section=source_section.strip(),
        generated_answer=generated_answer.strip(),
    )


def build_judge_messages(**kwargs: Any) -> list[dict[str, str]]:
    """Send the supplied judge prompt without adding a competing rubric."""
    return [{"role": "user", "content": make_judge_prompt(**kwargs)}]


def parse_judge_output(text: str) -> tuple[str, str]:
    """Parse exactly one VERDICT line and one REASON line."""
    verdicts: list[str] = []
    reasons: list[str] = []
    for raw_line in str(text or "").strip().splitlines():
        line = raw_line.strip()
        verdict_match = _VERDICT_LINE.match(line)
        if verdict_match:
            verdicts.append(verdict_match.group(1).upper())
            continue
        reason_match = _REASON_LINE.match(line)
        if reason_match:
            reasons.append(reason_match.group(1).strip())

    if len(verdicts) != 1:
        raise ValueError(f"expected exactly one VERDICT line, found {len(verdicts)}")
    if len(reasons) != 1 or not reasons[0]:
        raise ValueError(f"expected exactly one non-empty REASON line, found {len(reasons)}")
    return verdicts[0], reasons[0]


def _count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [row for row in rows if row.get("judge_verdict") in {"O", "X"}]
    correct = sum(row.get("judge_verdict") == "O" for row in graded)
    incorrect = sum(row.get("judge_verdict") == "X" for row in graded)
    return {
        "n": len(graded),
        "O": correct,
        "X": incorrect,
        "accuracy": correct / len(graded) if graded else 0.0,
        "judge_error_count": sum(bool(row.get("judge_error")) for row in rows),
    }


def summarize_judgements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate strict O/X results for paper-facing comparisons."""
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_strategy_budget: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        strategy = str(row.get("strategy") or "unknown")
        budget = row.get("context_token_budget")
        budget_label = "none" if budget is None else str(budget)
        by_strategy[strategy].append(row)
        by_strategy_budget[f"{strategy}@{budget_label}t"].append(row)
        by_type[str(row.get("qa_type") or "unknown")].append(row)
        by_section[str(row.get("source_section") or "unknown")].append(row)

    return {
        "overall": _count_rows(rows),
        "by_strategy": {key: _count_rows(value) for key, value in sorted(by_strategy.items())},
        "by_strategy_budget": {
            key: _count_rows(value) for key, value in sorted(by_strategy_budget.items())
        },
        "by_type": {key: _count_rows(value) for key, value in sorted(by_type.items())},
        "by_section": {key: _count_rows(value) for key, value in sorted(by_section.items())},
    }
