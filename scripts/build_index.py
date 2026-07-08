from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from kns_rag.io import load_jsonl, resolve_path
from kns_rag.retrieval import build_embeddings_for_chunks, save_index


def selected_strategies(cfg: dict, strategy_arg: str) -> list[str]:
    configured = cfg.get("chunking", {}).get("strategies") or []
    return configured if strategy_arg == "all" else [strategy_arg]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunks_dir = resolve_path(ROOT, cfg["paths"].get("chunks_dir", "data/chunks"))
    output_dir = resolve_path(ROOT, cfg["paths"].get("output_dir", "outputs"))
    index_root = output_dir / "indexes"
    model_name = cfg.get("embedding_model", {}).get("name")
    if not model_name:
        raise ValueError("config.embedding_model.name is required")

    for strategy in selected_strategies(cfg, args.strategy):
        chunk_path = chunks_dir / f"{strategy}.jsonl"
        if not chunk_path.exists():
            raise FileNotFoundError(f"missing chunks: {chunk_path}. Run scripts/build_chunks.py first.")
        chunks = load_jsonl(chunk_path)
        embeddings = build_embeddings_for_chunks(
            chunks,
            model_name=model_name,
            batch_size=args.batch_size,
        )
        out_dir = index_root / strategy
        save_index(out_dir, strategy=strategy, chunks=chunks, embeddings=embeddings, model_name=model_name)
        print(f"{strategy}: {len(chunks)} chunks indexed -> {out_dir}")


if __name__ == "__main__":
    main()
