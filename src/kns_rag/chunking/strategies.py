"""Chunking strategy implementations.

이 모듈은 순수 변환 로직만 담당한다. 파일 입출력과 CLI 인자는
scripts/build_chunks.py에서 처리한다.

출력 스키마는 모든 전략에서 동일하게 맞춘다.
"""

from __future__ import annotations

import re
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
    "semantic": {"target_chars": 900, "max_chars": 1200},
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


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in RE_SENTENCE_BOUNDARY.split(text) if s.strip()]


def semantic(
    records: list[dict[str, Any]],
    *,
    target_chars: int = 900,
    max_chars: int = 1200,
) -> list[dict[str, Any]]:
    """Lightweight semantic-ish chunking using sentence boundaries.

    This is a deterministic placeholder for the current pipeline stage. It keeps
    natural sentence boundaries instead of hard character cuts. If an embedding
    boundary detector is added later, it should preserve this output schema.
    """
    if target_chars <= 0 or max_chars <= 0:
        raise ValueError("target_chars and max_chars must be positive")
    if target_chars > max_chars:
        raise ValueError("target_chars must be <= max_chars")

    chunks = []
    for record in records:
        text = _record_text(record)
        base = _base_metadata(record)
        lco = base.get("lco")
        sentences = _split_sentences(text)
        if not sentences and text:
            sentences = [text]

        current: list[str] = []
        chunk_start = 0
        cursor = 0
        idx = 1

        def flush(end_cursor: int) -> None:
            nonlocal current, chunk_start, idx
            if not current:
                return
            body = " ".join(current).strip()
            metadata = {
                **base,
                "parent_id": record.get("id"),
                "chunk_type": "semantic_sentence_group",
                "chunk_index": idx,
                "char_start": chunk_start,
                "char_end": end_cursor,
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
            idx += 1
            current = []

        for sent in sentences:
            pos = text.find(sent, cursor)
            if pos < 0:
                pos = cursor
            sent_end = pos + len(sent)
            next_text = " ".join([*current, sent]).strip()
            if current and (len(next_text) > max_chars or len(" ".join(current)) >= target_chars):
                flush(pos)
                chunk_start = pos
            if not current:
                chunk_start = pos
            current.append(sent)
            cursor = sent_end
        flush(len(text))

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
