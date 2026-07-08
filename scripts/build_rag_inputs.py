from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from kns_rag.io import load_jsonl, resolve_path, write_jsonl


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--context-k", type=int, default=5)
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = resolve_path(ROOT, cfg["paths"].get("output_dir", "outputs"))
    runs_path = resolve_path(ROOT, args.runs) if args.runs else output_dir / "retrieval" / "runs.jsonl"
    out_path = resolve_path(ROOT, args.out) if args.out else output_dir / "generation" / "rag_inputs.jsonl"

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
