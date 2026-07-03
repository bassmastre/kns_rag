"""4개 청킹 전략. 별도 스크립트로 나누지 않고 동일 파이프라인에서
전략 이름으로 스위치되는 함수 형태로 유지 (config.chunking.strategies 참고).
"""

from .base import Chunk


def naive_fixed_length(text: str, doc_id: str, **params) -> list[Chunk]:
    raise NotImplementedError


def sliding_window(text: str, doc_id: str, **params) -> list[Chunk]:
    raise NotImplementedError


def semantic(text: str, doc_id: str, **params) -> list[Chunk]:
    """임베딩 유사도 기반 경계 탐지."""
    raise NotImplementedError


def hierarchical(text: str, doc_id: str, **params) -> list[Chunk]:
    """summary node + 2단계 검색."""
    raise NotImplementedError


STRATEGIES = {
    "naive_fixed_length": naive_fixed_length,
    "sliding_window": sliding_window,
    "semantic": semantic,
    "hierarchical": hierarchical,
}
