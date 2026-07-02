# 프로젝트: 원자력 규제문서 RAG 청킹 전략 비교 (KNS 추계 학술대회)

## 목적
NUREG-1431 Vol.1(STS)/Vol.2(Bases)를 대상으로 4가지 청킹 전략을
비교하는 짧은 실험 논문(1~5p, domestic conference) 코드 작성.
청킹 전략만이 유일한 독립변수. 그 외 모든 요소(임베딩 모델,
생성 LLM, judge LLM, retrieval 방식)는 고정.

## 확정된 설계 (변경 금지 — 임의로 확장하지 말 것)
- 대상 문서: NUREG-1431 Vol.1 + Vol.2, REVIEWER'S NOTE는 정규식으로 제거
- QA 소스 섹션: 3.4.1, 3.4.10, 3.4.13, 3.4.16 (Vol.1), 대응 Bases 섹션 (Vol.2)
  ※ 인덱싱은 전체 문서, QA 추출만 위 4개 섹션으로 한정
- 청킹 전략 4종: naive fixed-length / sliding window / semantic
  (임베딩 유사도 기반 경계 탐지) / hierarchical (summary node + 2단계 검색)
- Retrieval: Dense search 고정 (lexical/hybrid 비교는 별도 프로젝트, 이 논문 범위 아님)
- QA 데이터셋: 총 60~62개, 5개 유형
  (extractive ~22, condition-action mapping ~22, definition ~6,
  rationale/Vol.2 ~6, unanswerable ~6). 통계검정용 집계는 ~44개.
- 평가: retrieval(Hit@k, MRR) + LLM-as-judge(QA accuracy, groundedness/hallucination)
- Hit@k/MRR 판정: **span-level containment 기준(기본) + IoU(보조)**.
  절대 청크 단위로 gold answer를 태깅하지 말 것 — 원문 좌표계
  (doc_id, page, char_start, char_end) 기준으로 태깅.
- 결과 테이블에는 전략별 평균 청크 크기를 항상 병기 (크기 confound 명시용)

## 명시적으로 제외 (하지 말 것)
- Reranker 비교, ablation study, 멀티 임베딩/멀티 LLM 비교
- Retrieval 방식 비교(lexical vs dense vs hybrid) — 별도 프로젝트
- 위 4개 섹션 외 전체 문서에 대한 QA 자동 생성

## 하드웨어 제약
- RTX 3060 12GB 단일 GPU. 로컬 모델 사용 시 7~8B 양자화가 상한.
- Judge/Generator LLM을 로컬로 할지 API로 할지는 아직 미확정 —
  코드 작성 시 이 부분은 config로 분리해서 나중에 스위치 가능하게 할 것.

## 아직 미확정 (코드에서 하드코딩하지 말고 config/TODO로 남길 것)
- Generator LLM / Judge LLM: 로컬 vs API
- 임베딩 모델 종류
- PDF 표 구조(ACTIONS 표의 AND/OR indent) 추출 방식:
  pdfplumber extract_words() 좌표 기반 클러스터링으로 검증 예정,
  아직 실제 페이지 테스트 전.

## 코드 작성 원칙
- 전처리(원문 좌표 인덱싱)와 청킹은 완전히 분리된 모듈. 청킹 전략이
  바뀌어도 gold span 좌표는 불변이어야 함.
- 청킹 전략 4종은 동일한 출력 스키마(chunk_id, text,
  source_ref: {doc_id, char_start, char_end}, parent_id)를 따를 것.
- 실험 코드는 4개 전략에 대해 동일 파이프라인이 반복 실행되는 구조로 —
  전략별로 별도 스크립트를 만들지 말 것.