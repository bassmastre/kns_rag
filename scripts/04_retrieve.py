"""Stage 04: QA 질의 -> dense retrieval runs (outputs/retrieval/runs.jsonl).

이 스테이지부터는 data/qa/dddd.jsonl(사람 검증 QA)이 있어야 동작한다.
dddd.jsonl은 자동 생성물이 아니라 사람이 검증한 산출물이다 — 01~03과 달리
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
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Deprecated alias for --candidate-k.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Maximum cumulative content.body tokens retained per query.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.candidate_k is not None and args.top_k is not None:
        parser.error("use only one of --candidate-k and --top-k")

    cfg = load_config(args.config)

    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    if not qa_path.exists():
        raise FileNotFoundError(
            f"missing QA file: {qa_path}. Create JSONL with fields id, question, evidence_keywords."
        )
    qa_records = load_jsonl(qa_path)

    evaluation_cfg = cfg.raw.get("evaluation", {})
    token_budgets = [int(x) for x in (evaluation_cfg.get("token_budgets") or [])]
    if any(value <= 0 for value in token_budgets):
        raise ValueError("evaluation.token_budgets must contain positive integers")

    retrieval_cfg = cfg.raw.get("retrieval", {})
    candidate_k = (
        args.candidate_k
        or args.top_k
        or int(retrieval_cfg.get("candidate_k") or 100)
    )
    max_token_budget = args.token_budget or (max(token_budgets) if token_budgets else None)
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
            candidate_k=candidate_k,
            max_token_budget=max_token_budget,
            batch_size=args.batch_size,
        )
        all_runs.extend(runs)
        avg_selected = sum(run["selected_count"] for run in runs) / len(runs) if runs else 0.0
        avg_tokens = sum(run["selected_token_count"] for run in runs) / len(runs) if runs else 0.0
        print(
            f"{strategy}: retrieved {len(runs)} queries, "
            f"avg_chunks={avg_selected:.2f}, avg_tokens={avg_tokens:.1f}"
        )

    write_jsonl(out_path, all_runs)
    print(f"runs -> {out_path}")


if __name__ == "__main__":
    main()
