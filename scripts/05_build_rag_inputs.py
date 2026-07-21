"""Stage 05: retrieval runs -> downstream RAG inputs.

Stage 04의 runs.jsonl이 있어야 동작한다. 기본적으로 config의 모든
`evaluation.token_budgets`에 대해 동일 질의·전략 조합을 각각 생성한다.
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.downstream import build_answer_messages
from kns_rag.io import load_jsonl, write_jsonl


def select_contexts(
    results: list[dict],
    *,
    token_budget: int | None,
    context_k: int | None,
) -> list[dict]:
    contexts = []
    for result in results:
        cumulative_tokens = int(result.get("cumulative_tokens") or 0)
        if token_budget is not None and cumulative_tokens > token_budget:
            break
        contexts.append(result)
        if context_k is not None and len(contexts) >= context_k:
            break
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--token-budget",
        type=int,
        action="append",
        dest="token_budgets",
        help="Budget to include. Repeat for multiple budgets; defaults to config values.",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=None,
        help="Optional additional chunk-count cap. Disabled by default.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_path = cfg.resolve(args.runs) if args.runs else cfg.retrieval_runs_file
    out_path = cfg.resolve(args.out) if args.out else cfg.rag_inputs_file

    configured_budgets = [
        int(x) for x in (cfg.raw.get("evaluation", {}).get("token_budgets") or [])
    ]
    token_budgets = sorted(set(args.token_budgets or configured_budgets))
    if not token_budgets:
        raise ValueError("at least one token budget is required")
    if any(value <= 0 for value in token_budgets):
        raise ValueError("token budgets must be positive")

    rows = []
    for run in load_jsonl(runs_path):
        run_budget = run.get("max_token_budget")
        if run_budget is not None and max(token_budgets) > int(run_budget):
            raise ValueError(
                f"requested budget {max(token_budgets)} exceeds Stage 04 run budget {run_budget}; "
                "rerun scripts/04_retrieve.py with a larger --token-budget"
            )
        for token_budget in token_budgets:
            contexts = select_contexts(
                run.get("results") or [],
                token_budget=token_budget,
                context_k=args.context_k,
            )
            question = str(run.get("question") or "")
            qa_id = str(run.get("qa_id") or "")
            strategy = str(run.get("strategy") or "")
            experiment_id = f"{qa_id}::{strategy}::{token_budget}t"
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "qa_id": qa_id,
                    "qa_type": run.get("qa_type"),
                    "strategy": strategy,
                    "question": question,
                    "context_k": args.context_k,
                    "context_token_budget": token_budget,
                    "context_tokens": contexts[-1].get("cumulative_tokens", 0) if contexts else 0,
                    "context_count": len(contexts),
                    "contexts": contexts,
                    "messages": build_answer_messages(question, contexts),
                }
            )

    write_jsonl(out_path, rows)
    print(
        f"rag inputs -> {out_path} "
        f"({len(rows)} rows; budgets={','.join(map(str, token_budgets))})"
    )


if __name__ == "__main__":
    main()
