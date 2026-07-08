"""Stage 04: QA 질의 -> dense retrieval runs (outputs/retrieval/runs.jsonl).

이 스테이지부터는 data/qa/qa.jsonl(사람 검증 QA)이 있어야 동작한다.
qa.jsonl은 자동 생성물이 아니라 사람이 검증한 산출물이다 — 01~03과 달리
파이프라인이 만들어 주지 않는다. (스모크 테스트용은 scripts/qa_smoke.py 참고.)
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl
from kns_rag.retrieval import load_index, retrieve_queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'.")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    if not qa_path.exists():
        raise FileNotFoundError(
            f"missing QA file: {qa_path}. Create JSONL with fields id, question, gold_evidence_ids."
        )
    qa_records = load_jsonl(qa_path)

    k_values = cfg.raw.get("evaluation", {}).get("k_values") or [1, 3, 5]
    top_k = args.top_k or max(k_values)
    out_path = cfg.resolve(args.out) if args.out else cfg.retrieval_runs_file

    all_runs = []
    for strategy in cfg.selected_strategies(args.strategy):
        index_dir = cfg.index_dir(strategy)
        if not index_dir.exists():
            raise FileNotFoundError(
                f"missing index: {index_dir}. Run scripts/03_build_index.py first."
            )
        chunks, embeddings, meta = load_index(index_dir)
        model_name = meta.get("model_name") or cfg.raw.get("embedding_model", {}).get("name")
        runs = retrieve_queries(
            qa_records,
            chunks=chunks,
            chunk_embeddings=embeddings,
            model_name=model_name,
            strategy=strategy,
            top_k=top_k,
            batch_size=args.batch_size,
        )
        all_runs.extend(runs)
        print(f"{strategy}: retrieved {len(runs)} queries")

    write_jsonl(out_path, all_runs)
    print(f"runs -> {out_path}")


if __name__ == "__main__":
    main()
