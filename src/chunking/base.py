"""청킹 전략 공통 출력 스키마.

4개 전략(naive_fixed_length / sliding_window / semantic / hierarchical)은
모두 이 스키마를 따른다. 전략이 바뀌어도 gold span 좌표(원문 좌표계)는
불변이어야 하므로, source_ref는 전처리 단계에서 만든 좌표를 그대로 참조한다.
"""

from dataclasses import dataclass


@dataclass
class SourceRef:
    doc_id: str
    char_start: int
    char_end: int


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_ref: SourceRef
    parent_id: str | None = None
