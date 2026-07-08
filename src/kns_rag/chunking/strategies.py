"""Chunking strategy implementations.

이 모듈은 순수 변환 로직만 담당한다. 파일 입출력과 CLI 인자는
scripts/build_chunks.py에서 처리한다.

출력 스키마는 모든 전략에서 동일하게 맞춘다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

RE_ACTION_LABEL = re.compile(r"\b([A-Z]\.\d+(?:\.\d+)?)\b")
RE_WORD = re.compile(r"\S+")

RAW_INPUT = "raw"
ACTION_SOURCE_INPUT = "hierarchical_source"
CONDITION_CHUNKS_INPUT = "condition_chunks"


STRATEGY_INPUTS = {
    "naive_fixed_length": RAW_INPUT,
    "sliding_window": RAW_INPUT,
    "semantic": RAW_INPUT,
    "action_logic": ACTION_SOURCE_INPUT,
    "condition_aware": CONDITION_CHUNKS_INPUT,
    # Backward-compatible alias from the early config skeleton.
    "hierarchical": CONDITION_CHUNKS_INPUT,
}


DEFAULT_PARAMS = {
    "naive_fixed_length": {"chunk_size": 900},
    "sliding_window": {"chunk_size": 900, "overlap": 200},
    "semantic": {
        "model_name": None,
        "min_chars": 450,
        "target_chars": 900,
        "max_chars": 1200,
        "context_window_words": 45,
        "boundary_step_words": 3,
        "batch_size": 32,
    },
    "action_logic": {},
    "condition_aware": {},
    "hierarchical": {},
}


def _text_or_empty(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _record_text(record: dict[str, Any]) -> str:
    if "raw_text" in record:
        return _text_or_empty(record.get("raw_text"))
    return _text_or_empty(record.get("content", {}).get("body"))


def _evidence_ids_from_text(lco: str | None, text: str) -> list[str]:
    """Best-effort evidence ids for raw-derived chunks.

    Raw chunks do not carry structured action ids. We recover action labels that
    are explicitly present in the chunk text. This is intentionally conservative:
    it never infers missing actions from Condition labels.
    """
    if not lco:
        return []
    labels = list(dict.fromkeys(RE_ACTION_LABEL.findall(text)))
    evidence_ids = [f"{lco}/{label}" for label in labels]
    if "LCO" in text and f"{lco}/LCO" not in evidence_ids:
        evidence_ids.append(f"{lco}/LCO")
    return evidence_ids


def _base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    return {
        "lco": metadata.get("lco") or record.get("id"),
        "title": metadata.get("title"),
        "source_doc": metadata.get("source_doc"),
        "source_pages": metadata.get("source_pages"),
    }


def _make_chunk(
    *,
    strategy: str,
    chunk_id: str,
    body: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"{strategy}::{chunk_id}",
        "strategy": strategy,
        "metadata": metadata,
        "content": {"body": body},
    }


def _window_spans(text: str, chunk_size: int, overlap: int = 0) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not text:
        return []

    spans = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        start += step
    return spans


def naive_fixed_length(
    records: list[dict[str, Any]],
    *,
    chunk_size: int = 900,
) -> list[dict[str, Any]]:
    chunks = []
    for record in records:
        text = _record_text(record)
        base = _base_metadata(record)
        lco = base.get("lco")
        for idx, (start, end) in enumerate(_window_spans(text, chunk_size), 1):
            body = text[start:end].strip()
            metadata = {
                **base,
                "parent_id": record.get("id"),
                "chunk_type": "fixed_length",
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "evidence_ids": _evidence_ids_from_text(lco, body),
            }
            chunks.append(
                _make_chunk(
                    strategy="naive_fixed_length",
                    chunk_id=f"{record.get('id')}#{idx}",
                    body=body,
                    metadata=metadata,
                )
            )
    return chunks


def sliding_window(
    records: list[dict[str, Any]],
    *,
    chunk_size: int = 900,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    chunks = []
    for record in records:
        text = _record_text(record)
        base = _base_metadata(record)
        lco = base.get("lco")
        for idx, (start, end) in enumerate(_window_spans(text, chunk_size, overlap), 1):
            body = text[start:end].strip()
            metadata = {
                **base,
                "parent_id": record.get("id"),
                "chunk_type": "sliding_window",
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "overlap": overlap,
                "evidence_ids": _evidence_ids_from_text(lco, body),
            }
            chunks.append(
                _make_chunk(
                    strategy="sliding_window",
                    chunk_id=f"{record.get('id')}#{idx}",
                    body=body,
                    metadata=metadata,
                )
            )
    return chunks


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "semantic chunking requires sentence-transformers. "
            "Install it or build non-semantic strategies only."
        ) from exc
    return SentenceTransformer(model_name)


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in RE_WORD.finditer(text)]


def _cosine_similarities(embeddings: Any) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("semantic chunking requires numpy") from exc

    arr = np.asarray(embeddings, dtype="float32")
    if len(arr) < 2:
        return []
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return [float((arr[i] * arr[i + 1]).sum()) for i in range(len(arr) - 1)]


def _semantic_boundary_candidates(
    text: str,
    words: list[tuple[str, int, int]],
    *,
    context_window_words: int,
    boundary_step_words: int,
) -> tuple[list[int], list[str]]:
    """Build embedding contexts around candidate word boundaries.

    A boundary index i means a cut before words[i]. This function does not use
    sentence punctuation, list markers, condition labels, or parsed structure.
    It only proposes word-boundary positions and lets embeddings score semantic
    coherence across the boundary.
    """
    if context_window_words <= 0:
        raise ValueError("context_window_words must be positive")
    if boundary_step_words <= 0:
        raise ValueError("boundary_step_words must be positive")

    n = len(words)
    if n < 2:
        return [], []

    boundary_indices = list(range(1, n, boundary_step_words))
    if boundary_indices[-1] != n - 1:
        boundary_indices.append(n - 1)

    contexts = []
    for idx in boundary_indices:
        left_start_idx = max(0, idx - context_window_words)
        left_end_idx = idx - 1
        right_start_idx = idx
        right_end_idx = min(n - 1, idx + context_window_words - 1)

        left_start = words[left_start_idx][1]
        left_end = words[left_end_idx][2]
        right_start = words[right_start_idx][1]
        right_end = words[right_end_idx][2]

        contexts.append(text[left_start:left_end])
        contexts.append(text[right_start:right_end])

    return boundary_indices, contexts


def _semantic_boundary_scores(
    text: str,
    words: list[tuple[str, int, int]],
    model: Any,
    *,
    context_window_words: int,
    boundary_step_words: int,
    batch_size: int,
) -> dict[int, float]:
    boundary_indices, contexts = _semantic_boundary_candidates(
        text,
        words,
        context_window_words=context_window_words,
        boundary_step_words=boundary_step_words,
    )
    if not boundary_indices:
        return {}

    embeddings = model.encode(
        contexts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    pair_scores = _cosine_similarities(embeddings)

    # _cosine_similarities computes all adjacent pairs. We only need pairs
    # (0,1), (2,3), ... because contexts are [left0, right0, left1, right1, ...].
    scores = {}
    for i, boundary_idx in enumerate(boundary_indices):
        score_idx = i * 2
        if score_idx < len(pair_scores):
            scores[boundary_idx] = pair_scores[score_idx]
    return scores


def _first_boundary_after_char(
    words: list[tuple[str, int, int]],
    start_word_idx: int,
    char_limit: int,
) -> int:
    for idx in range(start_word_idx + 1, len(words)):
        if words[idx][1] >= char_limit:
            return idx
    return len(words)


def _build_semantic_spans_from_scores(
    words: list[tuple[str, int, int]],
    boundary_scores: dict[int, float],
    *,
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> list[tuple[int, int, int, int, float | None]]:
    """Build chunk spans by choosing the weakest semantic boundary.

    For each chunk, search candidate word boundaries in the valid range. Prefer
    boundaries after target_chars; choose the boundary with the lowest left/right
    context similarity. If there is no scored candidate, fall back to the nearest
    whitespace boundary around max_chars.
    """
    spans = []
    n = len(words)
    if n == 0:
        return spans

    start_word = 0
    while start_word < n:
        start_char = words[start_word][1]
        remaining_end = words[-1][2]
        if remaining_end - start_char <= max_chars:
            spans.append((start_word, n, start_char, remaining_end, None))
            break

        min_abs = start_char + min_chars
        target_abs = start_char + target_chars
        max_abs = start_char + max_chars

        valid = [
            idx
            for idx in boundary_scores
            if idx > start_word and min_abs <= words[idx][1] <= max_abs
        ]
        preferred = [idx for idx in valid if words[idx][1] >= target_abs]
        pool = preferred or valid

        if pool:
            cut_word = min(pool, key=lambda idx: boundary_scores[idx])
            cut_score = boundary_scores[cut_word]
        else:
            cut_word = _first_boundary_after_char(words, start_word, max_abs)
            cut_score = None

        if cut_word <= start_word:
            cut_word = min(start_word + 1, n)

        end_char = words[cut_word - 1][2]
        spans.append((start_word, cut_word, start_char, end_char, cut_score))
        start_word = cut_word

    return spans


def semantic(
    records: list[dict[str, Any]],
    *,
    model_name: str | None = None,
    min_chars: int = 450,
    target_chars: int = 900,
    max_chars: int = 1200,
    context_window_words: int = 45,
    boundary_step_words: int = 3,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Embedding-based semantic boundary chunking.

    This implementation does not split by sentence punctuation. It scores word
    boundary candidates by embedding the left and right local contexts around
    each boundary and selecting the lowest-coherence boundary within length
    constraints. It is therefore semantic-similarity based, while still not using
    regulatory condition/action parser output.
    """
    if not model_name:
        raise ValueError(
            "semantic chunking requires chunking.params.semantic.model_name "
            "or embedding_model.name in config.yaml"
        )
    if min_chars <= 0 or target_chars <= 0 or max_chars <= 0:
        raise ValueError("min_chars, target_chars, and max_chars must be positive")
    if min_chars > target_chars or target_chars > max_chars:
        raise ValueError("expected min_chars <= target_chars <= max_chars")

    model = _load_sentence_transformer(model_name)
    chunks = []

    for record in records:
        text = _record_text(record)
        if not text:
            continue
        base = _base_metadata(record)
        lco = base.get("lco")
        words = _word_spans(text)
        if not words:
            continue

        boundary_scores = _semantic_boundary_scores(
            text,
            words,
            model,
            context_window_words=context_window_words,
            boundary_step_words=boundary_step_words,
            batch_size=batch_size,
        )
        spans = _build_semantic_spans_from_scores(
            words,
            boundary_scores,
            min_chars=min_chars,
            target_chars=target_chars,
            max_chars=max_chars,
        )

        for idx, (word_start, word_end, start, end, cut_score) in enumerate(spans, 1):
            body = text[start:end].strip()
            metadata = {
                **base,
                "parent_id": record.get("id"),
                "chunk_type": "semantic_embedding_boundary",
                "semantic_method": "word_boundary_context_coherence",
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "word_start": word_start,
                "word_end": word_end,
                "embedding_model": model_name,
                "selected_boundary_similarity": cut_score,
                "context_window_words": context_window_words,
                "boundary_step_words": boundary_step_words,
                "evidence_ids": _evidence_ids_from_text(lco, body),
            }
            chunks.append(
                _make_chunk(
                    strategy="semantic",
                    chunk_id=f"{record.get('id')}#{idx}",
                    body=body,
                    metadata=metadata,
                )
            )

    return chunks


def action_logic(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize action-level logic-enriched source records to chunk schema."""
    chunks = []
    for record in records:
        body = _record_text(record)
        metadata = dict(record.get("metadata", {}))
        metadata["chunk_type"] = metadata.get("chunk_type") or "action_logic"
        metadata["evidence_ids"] = [record["id"]]
        chunks.append(
            _make_chunk(
                strategy="action_logic",
                chunk_id=record["id"],
                body=body,
                metadata=metadata,
            )
        )
    return chunks


def condition_aware(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize condition-level chunks to common chunk schema."""
    chunks = []
    for record in records:
        body = _record_text(record)
        metadata = dict(record.get("metadata", {}))
        metadata["chunk_type"] = metadata.get("chunk_type") or "condition"
        metadata["evidence_ids"] = metadata.get("evidence_ids") or [record["id"]]
        chunks.append(
            _make_chunk(
                strategy="condition_aware",
                chunk_id=record["id"],
                body=body,
                metadata=metadata,
            )
        )
    return chunks


def hierarchical(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible alias for condition_aware."""
    aliased = condition_aware(records)
    for chunk in aliased:
        chunk["strategy"] = "hierarchical"
        chunk["id"] = chunk["id"].replace("condition_aware::", "hierarchical::", 1)
    return aliased


STRATEGIES = {
    "naive_fixed_length": naive_fixed_length,
    "sliding_window": sliding_window,
    "semantic": semantic,
    "action_logic": action_logic,
    "condition_aware": condition_aware,
    "hierarchical": hierarchical,
}
