# kns_rag

원자력 규제문서 RAG에서 **청킹 전략 비교** 실험. 한국원자력학회 추계 학술대회(2026) 제출용.

## 한 줄 요지
원자력 STS 문서에서는 **문서 구조(condition-action 논리)를 인식하는 청킹**이,
도메인 무관한 정교한 기법(고정 길이·슬라이딩·의미 기반)보다 검색 성능이 낫다.
"더 정교한 청킹"이 아니라 "도메인 구조를 아는 청킹"이 이긴다는 것이 핵심 주장.

## 독립변수 = 청킹 전략 (유일)
1. **Naive** — 고정 길이
2. **Sliding** — 겹치는 윈도우
3. **Semantic** — 문장 임베딩 유사도로 경계 탐지
4. **Hierarchical (제안)** — STS 구조(condition 단위)를 경계로 분할

임베딩·generator·judge·retrieval·언어는 전부 고정. 청킹만 변수.

## 코퍼스
- **문서**: NUREG-1431 Vol.1 (STS) — 인덱싱은 3.4(RCS) 전체 (distractor 확보)
- **Gold QA 대상**: 3.4.1 / 3.4.10 / 3.4.13 / 3.4.16 4개 섹션
- **canonical source**: `hierarchical.jsonl`
  - `actions_text` (연속 줄글) → naive/sliding/semantic 입력
  - `condition_blocks` (구조) → hierarchical 입력
- `flat.jsonl` — condition-action 단위 파생 뷰 (dense baseline)

## 평가
- **Retrieval**: Hit@k, MRR (containment 기반, IoU 보조) + 평균 청크 크기·개수 병기
- **Generation**: LLM-as-judge (accuracy, groundedness)
- **Gold evidence**: flat chunk id 집합 (원문 좌표 아님)

## 왜 hierarchical이 이기나 (실험 가설)
- 조건의 조치들은 AND/OR로 묶임. 한 condition을 통째로 청킹하면 이 논리가 보존됨.
- Semantic 청킹은 B.1.1(grab sample)과 B.2.1(restore)을 의미가 다르다고 갈라놓지만,
  규제 논리상 한 조건의 필수 조치 쌍이라 갈라지면 안 됨 → 부분 회수 → 성능 저하.

## 구조
```
src/kns_rag/
  text.py      정제·정규화 + 라벨/커넥터
  layout.py    좌표 유틸 + 영역 분할
  parse.py     PDF 페이지 → raw dict
  corpus.py    섹션 병합 → hierarchical/flat
  (chunking/ retrieval/ eval/ 는 파싱 안정화 후 추가)
scripts/build_corpus.py
config.yaml    페이지 범위·컬럼 경계·모델명
```

## 제약
RTX 3060 12GB (임베딩 소형 + generator 7~8B 4-bit 양자화 or API / judge는 API 권장) ·
논문 1~5p · 기간 ~1개월

## 현황
- [x] 3.4.15 파싱 확정 (connector·옵션 액션·NOTE·LCO statement 정합)
- [ ] 파싱 모듈화 (codex)
- [ ] 3.4 전체 추출 (gold=정밀, distractor=줄글 fallback)
- [ ] 임베딩/generator/judge 확정
- [ ] QA 60여 개 (5 타입) + 4전략 청킹 → 평가
- [ ] Vol.2 Bases (rationale QA용, 후순위)