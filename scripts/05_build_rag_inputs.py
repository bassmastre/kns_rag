"""Stage 05: retrieval runs -> 생성용 프롬프트 (outputs/generation/rag_inputs.jsonl).

Stage 04의 runs.jsonl이 있어야 동작한다 (즉 data/qa/qa.jsonl 이후 경계).
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl


def make_prompt(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{i}] {ctx.get('body', '').strip()}" for i, ctx in enumerate(contexts, 1)
    )
    return (
        "Answer the question using only the provided context. "
        "If the answer is not supported by the context, say that it is not answerable from the provided context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )


def select_contexts(results: list[dict], *, token_budget: int | None, context_k: int | None) -> list[dict]:
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
        default=None,
        help="Context token budget. Defaults to the largest evaluation.token_budgets value.",
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

    token_budgets = [int(x) for x in (cfg.raw.get("evaluation", {}).get("token_budgets") or [])]
    token_budget = args.token_budget or (max(token_budgets) if token_budgets else None)

    rows = []
    for run in load_jsonl(runs_path):
        contexts = select_contexts(
            run.get("results") or [],
            token_budget=token_budget,
            context_k=args.context_k,
        )
        rows.append(
            {
                "qa_id": run.get("qa_id"),
                "strategy": run.get("strategy"),
                "question": run.get("question"),
                "context_k": args.context_k,
                "context_token_budget": token_budget,
                "context_tokens": contexts[-1].get("cumulative_tokens", 0) if contexts else 0,
                "contexts": contexts,
                "prompt": make_prompt(str(run.get("question") or ""), contexts),
            }
        )

    write_jsonl(out_path, rows)
    print(f"rag inputs -> {out_path}")


if __name__ == "__main__":
    main()
