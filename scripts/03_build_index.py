"""Stage 03: 청크 -> dense index (outputs/indexes/<strategy>/).

zero-arg 실행 가능. 01~03은 사람 개입 없이 순서대로 실행된다.
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl
from kns_rag.retrieval import build_embeddings_for_chunks, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--strategy", default="all", help="Strategy name or 'all'.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg.raw.get("embedding_model", {}).get("name")
    if not model_name:
        raise ValueError("config.embedding_model.name is required")

    for strategy in cfg.selected_strategies(args.strategy):
        chunk_path = cfg.chunks_file(strategy)
        if not chunk_path.exists():
            raise FileNotFoundError(
                f"missing chunks: {chunk_path}. Run scripts/02_build_chunks.py first."
            )
        chunks = load_jsonl(chunk_path)
        embeddings = build_embeddings_for_chunks(
            chunks,
            model_name=model_name,
            batch_size=args.batch_size,
        )
        out_dir = cfg.index_dir(strategy)
        save_index(out_dir, strategy=strategy, chunks=chunks, embeddings=embeddings, model_name=model_name)
        print(f"{strategy}: {len(chunks)} chunks indexed -> {out_dir}")


if __name__ == "__main__":
    main()
