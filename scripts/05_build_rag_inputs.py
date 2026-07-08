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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--context-k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs_path = cfg.resolve(args.runs) if args.runs else cfg.retrieval_runs_file
    out_path = cfg.resolve(args.out) if args.out else cfg.rag_inputs_file

    rows = []
    for run in load_jsonl(runs_path):
        contexts = (run.get("results") or [])[: args.context_k]
        rows.append(
            {
                "qa_id": run.get("qa_id"),
                "strategy": run.get("strategy"),
                "question": run.get("question"),
                "context_k": args.context_k,
                "contexts": contexts,
                "prompt": make_prompt(str(run.get("question") or ""), contexts),
            }
        )

    write_jsonl(out_path, rows)
    print(f"rag inputs -> {out_path}")


if __name__ == "__main__":
    main()
