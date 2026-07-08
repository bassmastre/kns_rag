from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kns_rag.io import load_jsonl, resolve_path, write_jsonl
from kns_rag.retrieval import load_index, retrieve_queries


def selected_strategies(cfg: dict, strategy_arg: str) -> list[str]:
    configured = cfg.get("chunking", {}).get("strategies") or []
    return configured if strategy_arg == "all" else [strategy_arg]


def default_qa_file(cfg: dict) -> Path:
    qa_dir = resolve_path(ROOT, cfg["paths"].get("qa_dir", "data/qa"))
    return qa_dir / "qa.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'.")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qa_path = resolve_path(ROOT, args.qa_file) if args.qa_file else default_qa_file(cfg)
    if not qa_path.exists():
        raise FileNotFoundError(
            f"missing QA file: {qa_path}. Create JSONL with fields id, question, gold_evidence_ids."
        )
    qa_records = load_jsonl(qa_path)

    k_values = cfg.get("evaluation", {}).get("k_values") or [1, 3, 5]
    top_k = args.top_k or max(k_values)
    output_dir = resolve_path(ROOT, cfg["paths"].get("output_dir", "outputs"))
    index_root = output_dir / "indexes"
    out_path = resolve_path(ROOT, args.out) if args.out else output_dir / "retrieval" / "runs.jsonl"

    all_runs = []
    for strategy in selected_strategies(cfg, args.strategy):
        index_dir = index_root / strategy
        if not index_dir.exists():
            raise FileNotFoundError(f"missing index: {index_dir}. Run scripts/build_index.py first.")
        chunks, embeddings, meta = load_index(index_dir)
        model_name = meta.get("model_name") or cfg.get("embedding_model", {}).get("name")
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
