"""원문 좌표 인덱싱.

전처리(extract_sections)와 청킹은 완전히 분리된 모듈이며, 이 모듈이 그
경계를 담당한다: (doc_id, page, char_start, char_end) 좌표계를 만들고,
gold answer span 태깅과 Hit@k/MRR 판정(span-level containment 기준,
IoU 보조)이 이 좌표계 위에서 이루어지도록 한다.
"""

# TODO: pdfplumber extract_words() 좌표 기반 클러스터링 검증 후 구현
