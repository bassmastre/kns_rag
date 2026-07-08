# kns_rag

원자력 규제문서(NUREG-1431 STS) RAG에서 **청킹 전략 비교** 실험.
한국원자력학회 추계 학술대회(2026) 제출용 단편 실험 논문(1~5p).

연구 배경·가설·설계 근거는 [`INFO.md`](INFO.md)를 참고. 이 문서는 리포의
현재 구현 상태와 실행 방법만 다룬다.

---

## 한 줄 요지

STS 문서에서는 **문서 구조(condition–action 논리)를 인식하는 청킹**이
도메인 무관 기법(고정 길이·슬라이딩·의미 기반)보다 검색 성능이 낫다.
"더 정교한 청킹"이 아니라 "도메인 구조를 아는 청킹"이 이긴다는 것이 핵심 주장.

## 독립변수 = 청킹 전략

| 전략 | 설명 | 입력 소스 |
|------|------|-----------|
| `naive_fixed_length` | 고정 길이 | `raw.jsonl`의 flattened `raw_text` |
| `sliding_window` | 겹치는 윈도우 | `raw.jsonl`의 flattened `raw_text` |
| `semantic` | 임베딩 기반 word-boundary coherence 경계 | `raw.jsonl`의 flattened `raw_text` |
| `action_logic` | action 단위 구조 청크 | `hierarchical_source.jsonl` |
| `condition_aware` | condition 단위 구조 청크 | `condition_chunks.jsonl` |

임베딩·retrieval·언어는 고정하고 청킹 전략만 비교한다. `evidence_ids`는 평가용
metadata이며 임베딩 대상은 항상 `content.body`뿐이다.

---

## 현재 상태 (2026-07 기준)

**완료**
- PDF 파싱 파이프라인(`parse.py` + `layout.py` + `text.py` + `corpus.py`)
- 코퍼스 산출물 생성(`build_corpus.py`)
- 5개 청킹 전략 생성(`build_chunks.py`)
- dense index 생성(`build_index.py`)
- retrieval run 생성(`retrieve.py`)
- RAG prompt 입력 생성(`build_rag_inputs.py`)
- retrieval metric 평가(`eval_retrieval.py`)
- end-to-end runner(`run_pipeline.py`)

**미완**
- QA 데이터셋 작성
- 실제 Generator 호출 / LLM-as-judge 채점
- Vol.2 Bases 통합

---

## 코퍼스 산출물

### `data/processed/`

| 파일 | 단위 | 용도 |
|------|------|------|
| `raw.jsonl` | LCO 조항당 1레코드 | naive/sliding/semantic 입력 |
| `hierarchical_source.jsonl` | action 단위 flat record | action_logic 입력 / gold evidence 기준 |
| `condition_chunks.jsonl` | condition 단위 record | condition_aware 입력 |

### `data/chunks/`

`build_chunks.py` 실행 후 전략별 청크 파일이 생성된다.

```text
naive_fixed_length.jsonl
sliding_window.jsonl
semantic.jsonl
action_logic.jsonl
condition_aware.jsonl
```

### `outputs/`

```text
outputs/indexes/<strategy>/chunks.jsonl
outputs/indexes/<strategy>/embeddings.npy
outputs/indexes/<strategy>/meta.json
outputs/retrieval/runs.jsonl
outputs/generation/rag_inputs.jsonl
outputs/eval/retrieval_metrics.json
```

---

## QA 파일 형식

기본 경로는 `data/qa/qa.jsonl`이다. 최소 필드는 다음과 같다.

```json
{"id":"q001","type":"condition_action_mapping","question":"What is required when one required RCS loop is not in operation with Rod Control System capable of rod withdrawal?","gold_evidence_ids":["3.4.5/C.1","3.4.5/C.2"],"answerable":true}
```

허용되는 gold 필드명은 `gold_evidence_ids`, `evidence_ids`, `gold_ids`,
또는 `gold.evidence_ids`이다. `type`이 `unanswerable`이면 retrieval metric에서는
기본적으로 제외된다.

---

## 실행

의존성:

```bash
pip install pdfplumber pyyaml numpy sentence-transformers
```

전체 retrieval 파이프라인:

```bash
python scripts/run_pipeline.py --config config.yaml --from-stage corpus --to-stage eval
```

청크까지 이미 만들었다면:

```bash
python scripts/run_pipeline.py --config config.yaml --from-stage index --to-stage eval
```

단계별 실행:

```bash
python scripts/build_corpus.py --config config.yaml
python scripts/build_chunks.py --config config.yaml --strategy all
python scripts/build_index.py --config config.yaml --strategy all
python scripts/retrieve.py --config config.yaml --strategy all --qa-file data/qa/qa.jsonl
python scripts/build_rag_inputs.py --config config.yaml
python scripts/eval_retrieval.py --config config.yaml --qa-file data/qa/qa.jsonl
```

특정 전략만 실행:

```bash
python scripts/build_chunks.py --config config.yaml --strategy condition_aware
python scripts/build_index.py --config config.yaml --strategy condition_aware
python scripts/retrieve.py --config config.yaml --strategy condition_aware --qa-file data/qa/qa.jsonl
```

파싱 결과 확인 예시:

```bash
python -c "import json; print(json.loads(open('data/processed/raw.jsonl', encoding='utf-8').readline())['raw_text'])"
```

Completion Time 매핑 점검 예시(CMD 한 줄):

```cmd
python -c "import json; p='data\\processed\\condition_chunks.jsonl'; bad=[]; [bad.append((i,r.get('id'),a.get('label'),ct)) for i,line in enumerate(open(p,encoding='utf-8'),1) for r in [json.loads(line)] for a in r.get('content',{}).get('actions',[]) for ct in [a.get('completion_time') or ''] if ct.startswith(('OR ','AND ','In accordance '))]; print('bad count:',len(bad)); [print(x) for x in bad]"
```

> `data/`, `outputs/`는 `.gitignore` 대상 — 산출물은 로컬 전용, 재실행 시 재생성.

---

## 코드 구조

```text
src/kns_rag/
  text.py          정제·정규화 + 라벨/커넥터 정규식
  layout.py        좌표 유틸 + 페이지 영역 분할(narr/tbl)
  parse.py         PDF 페이지 → raw dict (컬럼 밴드 파싱)
  corpus.py        섹션 병합 → raw/hierarchical/condition source 생성
  chunking/        청킹 전략 구현
  embeddings.py    SentenceTransformer 임베딩 유틸
  retrieval.py     dense index / retrieval 유틸
  evaluation.py    Hit@k / MRR 평가 유틸
scripts/
  build_corpus.py
  build_chunks.py
  build_index.py
  retrieve.py
  build_rag_inputs.py
  eval_retrieval.py
  run_pipeline.py
config.yaml
CLAUDE.md
```

---

## 제약

- **HW**: RTX 3060 12GB 1장. 임베딩은 소형(sentence-transformers 계열),
  generator는 7~8B 4-bit 양자화 또는 API, judge는 API 권장. 대규모 파인튜닝 배제.
- **분량**: 1~5p. **기간**: ~1개월.
- **환경**: conda env `kns_rag`, Python 3.11.
