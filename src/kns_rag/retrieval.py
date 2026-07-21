from __future__ import annotations

from pathlib import Path
from typing import Any

from .embeddings import count_text_tokens, encode_texts
from .io import chunk_body, load_json, load_jsonl, write_json, write_jsonl


def build_embeddings_for_chunks(chunks: list[dict[str, Any]], *, model_name: str, batch_size: int = 32) -> Any:
    """Embed only content.body. Metadata is not embedded."""
    texts = [chunk_body(c) for c in chunks]
    return encode_texts(texts, model_name=model_name, batch_size=batch_size, normalize=True, show_progress_bar=True)


def save_index(index_dir: Path, *, strategy: str, chunks: list[dict[str, Any]], embeddings: Any, model_name: str) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for dense retrieval") from exc

    index_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(index_dir / "chunks.jsonl", chunks)
    np.save(index_dir / "embeddings.npy", embeddings)
    dim = int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0
    write_json(index_dir / "meta.json", {
        "strategy": strategy,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": dim,
        "text_field": "content.body",
        "metadata_indexed": False,
    })


def load_index(index_dir: Path) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for dense retrieval") from exc

    chunks = load_jsonl(index_dir / "chunks.jsonl")
    embeddings = np.load(index_dir / "embeddings.npy")
    meta = load_json(index_dir / "meta.json")
    return chunks, embeddings, meta


def search_embeddings(query_embedding: Any, chunk_embeddings: Any, *, top_k: int) -> list[tuple[int, float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for dense retrieval") from exc

    if top_k <= 0 or len(chunk_embeddings) == 0:
        return []
    q = np.asarray(query_embedding, dtype="float32").reshape(-1)
    scores = chunk_embeddings @ q
    k = min(top_k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def select_ranked_prefix_by_token_budget(
    ranked_pairs: list[tuple[int, float]],
    *,
    token_counts: list[int],
    max_token_budget: int | None,
) -> list[tuple[int, float, int, int]]:
    """Keep the highest-ranked prefix that fits within the token budget.

    The selector never skips an oversized higher-ranked chunk to fit a smaller
    lower-ranked chunk. This strict-prefix rule preserves dense ranking order
    and avoids introducing a small-chunk selection bias.
    """
    if max_token_budget is not None and max_token_budget <= 0:
        raise ValueError("max_token_budget must be positive")

    selected: list[tuple[int, float, int, int]] = []
    cumulative_tokens = 0
    for chunk_idx, score in ranked_pairs:
        token_count = int(token_counts[chunk_idx])
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        next_total = cumulative_tokens + token_count
        if max_token_budget is not None and next_total > max_token_budget:
            break
        cumulative_tokens = next_total
        selected.append((chunk_idx, score, token_count, cumulative_tokens))
    return selected


def retrieve_queries(
    qa_records: list[dict[str, Any]],
    *,
    chunks: list[dict[str, Any]],
    chunk_embeddings: Any,
    model_name: str,
    strategy: str,
    candidate_k: int,
    max_token_budget: int | None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")

    questions = [str(q.get("question") or "").strip() for q in qa_records]
    query_embeddings = encode_texts(
        questions,
        model_name=model_name,
        batch_size=batch_size,
        normalize=True,
        show_progress_bar=False,
    )
    token_counts = count_text_tokens([chunk_body(chunk) for chunk in chunks], model_name=model_name)

    runs: list[dict[str, Any]] = []
    for qa, query_emb in zip(qa_records, query_embeddings):
        candidates = search_embeddings(query_emb, chunk_embeddings, top_k=candidate_k)
        selected = select_ranked_prefix_by_token_budget(
            candidates,
            token_counts=token_counts,
            max_token_budget=max_token_budget,
        )

        ranked = []
        for rank, (chunk_idx, score, token_count, cumulative_tokens) in enumerate(selected, 1):
            chunk = chunks[chunk_idx]
            ranked.append({
                "rank": rank,
                "score": score,
                "chunk_id": chunk.get("id"),
                "evidence_ids": chunk.get("metadata", {}).get("evidence_ids") or [],
                "body": chunk_body(chunk),
                "metadata": chunk.get("metadata", {}),
                "token_count": token_count,
                "cumulative_tokens": cumulative_tokens,
            })

        runs.append({
            "qa_id": qa.get("id"),
            "question": qa.get("question"),
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "strategy": strategy,
            "candidate_k": candidate_k,
            "max_token_budget": max_token_budget,
            "selected_count": len(ranked),
            "selected_token_count": ranked[-1]["cumulative_tokens"] if ranked else 0,
            "token_count_field": "content.body",
            "tokenizer_model": model_name,
            "selection_policy": "strict_ranked_prefix",
            "results": ranked,
        })
    return runs
