from __future__ import annotations

import json
from typing import Any


ANSWER_REQUIREMENTS = """Answer requirements:
- State the applicable LCO section number explicitly.
- Preserve every required AND-bound action or condition.
- Include every OR-alternative requested by the question.
- Preserve Condition letters, numeric limits, and Completion Times exactly.
- Do not add unsupported values, conditions, or actions.
- If you cannot support a complete answer, output exactly INSUFFICIENT_CONTEXT."""


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
REASON: <one sentence, citing which rule triggered the verdict>"""


def make_rag_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[{i}] {str(ctx.get('body') or '').strip()}" for i, ctx in enumerate(contexts, 1)
    )
    return (
        "Answer the question using only the provided context.\n\n"
        f"{ANSWER_REQUIREMENTS}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )


def make_llm_only_prompt(question: str) -> str:
    return (
        "Answer this nuclear plant Technical Specifications question from the model's own knowledge. "
        "No retrieved context is provided.\n\n"
        f"{ANSWER_REQUIREMENTS}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def make_judge_prompt(
    *,
    question: str,
    reference_answer: str,
    evidence_keywords: list[list[str]],
    source_section: str,
    generated_answer: str,
) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question.strip(),
        reference_answer=reference_answer.strip(),
        evidence_keywords=json.dumps(evidence_keywords, ensure_ascii=False),
        source_section=source_section.strip(),
        generated_answer=generated_answer.strip(),
    )
