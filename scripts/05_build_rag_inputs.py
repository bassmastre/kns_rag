"""Stage 05: retrieval runs -> 생성용 프롬프트 (outputs/generation/rag_inputs.jsonl).

Stage 04의 runs.jsonl이 있어야 동작한다 (즉 data/qa/dddd.jsonl 이후 경계).
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl
from kns_rag.prompts import make_rag_prompt


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
        question = str(run.get("question") or "")
        rows.append(
            {
                "qa_id": run.get("qa_id"),
                "qa_type": run.get("qa_type"),
                "strategy": run.get("strategy"),
                "question": question,
                "context_k": args.context_k,
                "context_count": len(contexts),
                "contexts": contexts,
                "prompt": make_rag_prompt(question, contexts),
            }
        )

    write_jsonl(out_path, rows)
    print(f"rag inputs -> {out_path}")


if __name__ == "__main__":
    main()
