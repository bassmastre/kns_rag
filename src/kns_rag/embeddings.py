from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for dense retrieval. "
            "Install it in the active conda environment."
        ) from exc
    return SentenceTransformer(model_name)


def count_text_tokens(texts: list[str], *, model_name: str) -> list[int]:
    """Count content tokens with the embedding model tokenizer.

    Special tokens are excluded because the retrieval budget is applied to the
    concatenated chunk bodies, not to separately encoded model inputs.
    """
    model = load_sentence_transformer(model_name)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(f"embedding model does not expose a tokenizer: {model_name}")

    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None or len(input_ids) != len(texts):
        raise RuntimeError("tokenizer returned an unexpected input_ids shape")
    return [len(ids) for ids in input_ids]


def encode_texts(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> Any:
    """Encode texts with SentenceTransformer and return a float32 numpy array."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for dense retrieval") from exc

    model = load_sentence_transformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )
    arr = np.asarray(embeddings, dtype="float32")
    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
    return arr
