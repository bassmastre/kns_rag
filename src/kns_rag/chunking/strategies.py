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
RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

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
        "similarity_percentile": 20,
        "similarity_threshold": None,
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


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentence-like units and keep char spans.

    PDF-extracted regulatory text is not perfectly punctuated. This function uses
    punctuation boundaries when available and falls back to one unit for the whole
    text if no boundary is detected.
    """
    spans: list[tuple[str, int, int]] = []
    start = 0
    for part in RE_SENTENCE_BOUNDARY.split(text):
        sent = part.strip()
        if not sent:
            start += len(part)
            continue
        pos = text.find(sent, start)
        if pos < 0:
            pos = start
        end = pos + len(sent)
        spans.append((sent, pos, end))
        start = end
    if not spans and text.strip():
        stripped = text.strip()
        pos = text.find(stripped)
        spans.append((stripped, pos, pos + len(stripped)))
    return spans


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


def _semantic_cut_threshold(
    similarities: list[float],
    *,
    similarity_percentile: float,
    similarity_threshold: float | None,
) -> float | None:
    if not similarities:
        return None
    if similarity_threshold is not None:
        return float(similarity_threshold)
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("semantic chunking requires numpy") from exc
    return float(np.percentile(similarities, similarity_percentile))


def _build_semantic_spans(
    units: list[tuple[str, int, int]],
    similarities: list[float],
    *,
    min_chars: int,
    target_chars: int,
    max_chars: int,
    threshold: float | None,
) -> list[tuple[int, int, int, int, list[float]]]:
    """Return chunk spans from sentence units using adjacent similarity drops.

    Output tuple:
    (unit_start_idx, unit_end_exclusive, char_start, char_end, boundary_scores_inside)
    """
    if not units:
        return []

    spans = []
    start_unit = 0
    start_char = units[0][1]
    internal_scores: list[float] = []

    for i in range(len(units) - 1):
        current_end = units[i][2]
        current_len = current_end - start_char
        sim = similarities[i]
        internal_scores.append(sim)

        low_similarity = threshold is not None and sim <= threshold
        reached_target = current_len >= target_chars
        reached_max = units[i + 1][2] - start_char > max_chars
        can_cut = current_len >= min_chars

        if reached_max or (can_cut and low_similarity and reached_target):
            spans.append((start_unit, i + 1, start_char, current_end, internal_scores[:-1]))
            start_unit = i + 1
            start_char = units[start_unit][1]
            internal_scores = []

    spans.append((start_unit, len(units), start_char, units[-1][2], internal_scores))
    return spans


def semantic(
    records: list[dict[str, Any]],
    *,
    model_name: str | None = None,
    min_chars: int = 450,
    target_chars: int = 900,
    max_chars: int = 1200,
    similarity_percentile: float = 20,
    similarity_threshold: float | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Embedding-based semantic boundary chunking.

    Algorithm:
    1. Split each raw LCO text into sentence-like units.
    2. Embed units with SentenceTransformer.
    3. Compute adjacent-unit cosine similarities.
    4. Cut at low-similarity boundaries, while respecting min/target/max chars.

    This is still a baseline: it uses semantic similarity only, not regulatory
    condition/action structure.
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
    if not (0 <= similarity_percentile <= 100):
        raise ValueError("similarity_percentile must be between 0 and 100")

    model = _load_sentence_transformer(model_name)
    chunks = []

    for record in records:
        text = _record_text(record)
        if not text:
            continue
        base = _base_metadata(record)
        lco = base.get("lco")
        units = _split_sentences_with_spans(text)
        if not units:
            continue

        unit_texts = [u[0] for u in units]
        embeddings = model.encode(
            unit_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        similarities = _cosine_similarities(embeddings)
        threshold = _semantic_cut_threshold(
            similarities,
            similarity_percentile=similarity_percentile,
            similarity_threshold=similarity_threshold,
        )
        spans = _build_semantic_spans(
            units,
            similarities,
            min_chars=min_chars,
            target_chars=target_chars,
            max_chars=max_chars,
            threshold=threshold,
        )

        for idx, (unit_start, unit_end, start, end, scores) in enumerate(spans, 1):
            body = text[start:end].strip()
            metadata = {
                **base,
                "parent_id": record.get("id"),
                "chunk_type": "semantic_embedding_boundary",
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
                "unit_start": unit_start,
                "unit_end": unit_end,
                "embedding_model": model_name,
                "similarity_threshold": threshold,
                "mean_internal_similarity": (sum(scores) / len(scores)) if scores else None,
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
