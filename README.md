# kns_rag

원자력 규제문서(NUREG-1431 STS) RAG에서 **청킹 전략 비교** 실험.
한국원자력학회 추계 학술대회(2026) 제출용 단편 실험 논문(1~5p).

연구 배경·가설·설계 근거는 [`INFO.md`](INFO.md)를 참고한다. 이 문서는
리포의 현재 구현 상태와 실행 방법만 다룬다.

---

## 한 줄 요지

STS 문서에서는 **문서 구조(condition–action 논리)를 인식하는 청킹**이
도메인 무관 기법(고정 길이·슬라이딩·의미 기반)보다 검색 성능이 낫다는
가설을 검증한다. 핵심은 "더 정교한 청킹"이 아니라 **"도메인 구조를 아는
청킹"**이다.

---

## 전략 정의

| 전략 | 역할 | 입력 소스 | 비고 |
|------|------|-----------|------|
| `naive_fixed_length` | baseline | `raw.jsonl`의 flattened `raw_text` | 고정 길이 |
| `sliding_window` | baseline | `raw.jsonl`의 flattened `raw_text` | overlap 포함 |
| `semantic` | baseline | `raw.jsonl`의 flattened `raw_text` | embedding boundary |
| `action_logic` | 구조 ablation | `hierarchical_source.jsonl` | action 단위 |
| `condition_aware` | 제안 전략 | `condition_chunks.jsonl` | condition 단위 |

고정 조건:

- retrieval method: dense
- embedding model: `config.yaml`의 `embedding_model.name`
- embedding text field: **`content.body`만 사용**
- `metadata.evidence_ids`: retrieval input이 아니라 evaluation label

---

## 현재 구현 상태

### 완료

- PDF parsing / section merge
- processed corpus 생성
- 5개 청킹 전략 생성
- dense index 생성
- retrieval run 생성
- retrieval metric 평가
- RAG prompt 입력 생성
- end-to-end runner

### 미완

- 최종 QA dataset 작성
- 실제 generator 호출
- LLM-as-judge 채점
- Vol.2 Bases 통합

---

## 산출물 구조

### `data/processed/`

| 파일 | 단위 | 용도 |
|------|------|------|
| `raw.jsonl` | LCO 조항당 1 record | raw-text baseline 입력 |
| `hierarchical_source.jsonl` | action-level flat record | `action_logic` 입력 / gold evidence 기준 |
| `condition_chunks.jsonl` | condition-level record | `condition_aware` 입력 |

### `data/chunks/`

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

## 실행

의존성:

```bash
pip install pdfplumber pyyaml numpy sentence-transformers
```

### QA 없이 실행 가능한 단계

코퍼스부터 인덱스까지는 QA 파일 없이 실행 가능하다.

```bash
python scripts/run_pipeline.py --config config.yaml --from-stage corpus --to-stage index
```

청크가 이미 있으면 index만 다시 생성한다.

```bash
python scripts/build_index.py --config config.yaml --strategy all
```

### QA 파일이 필요한 단계

아래 단계부터는 `data/qa/qa.jsonl` 또는 `--qa-file`로 지정한 QA 파일이 필요하다.

```bash
python scripts/retrieve.py --config config.yaml --strategy all --qa-file data/qa/qa.jsonl
python scripts/build_rag_inputs.py --config config.yaml
python scripts/eval_retrieval.py --config config.yaml --qa-file data/qa/qa.jsonl
```

`run_pipeline.py`에서 `retrieve`, `rag_inputs`, `eval`을 포함하면 QA 파일이 필요하다.

```bash
python scripts/run_pipeline.py --config config.yaml --from-stage index --to-stage eval --qa-file data/qa/qa.jsonl
```

### 단계별 실행

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

---

## QA 파일 최소 스키마

```json
{"id":"q001","type":"condition_action_mapping","question":"What is required when one required RCS loop is not in operation with Rod Control System capable of rod withdrawal?","gold_evidence_ids":["3.4.5/C.1","3.4.5/C.2"],"answerable":true}
```

허용 gold 필드명:

- `gold_evidence_ids`
- `evidence_ids`
- `gold_ids`
- `gold.evidence_ids`

`type`이 `unanswerable`이면 retrieval metric에서는 기본적으로 제외한다.

---

## 점검 명령

파싱 결과 확인:

```bash
python -c "import json; print(json.loads(open('data/processed/raw.jsonl', encoding='utf-8').readline())['raw_text'])"
```

Completion Time 매핑 점검(CMD 한 줄):

```cmd
python -c "import json; p='data\\processed\\condition_chunks.jsonl'; bad=[]; [bad.append((i,r.get('id'),a.get('label'),ct)) for i,line in enumerate(open(p,encoding='utf-8'),1) for r in [json.loads(line)] for a in r.get('content',{}).get('actions',[]) for ct in [a.get('completion_time') or ''] if ct.startswith(('OR ','AND ','In accordance '))]; print('bad count:',len(bad)); [print(x) for x in bad]"
```

> `data/`, `outputs/`는 `.gitignore` 대상이다. 산출물은 로컬에서 재생성한다.

---

## 코드 구조

```text
src/kns_rag/
  text.py          정제·정규화 + 라벨/커넥터 정규식
  layout.py        좌표 유틸 + 페이지 영역 분할
  parse.py         PDF page → raw dict
  corpus.py        section merge → processed corpus
  chunking/        청킹 전략 구현
  embeddings.py    SentenceTransformer embedding utility
  retrieval.py     dense index / retrieval utility
  evaluation.py    Hit@k / MRR evaluation utility
scripts/
  build_corpus.py
  build_chunks.py
  build_index.py
  retrieve.py
  build_rag_inputs.py
  eval_retrieval.py
  run_pipeline.py
config.yaml
INFO.md
CLAUDE.md
```

---

## 제약

- HW: RTX 3060 12GB 1장 기준
- embedding: small sentence-transformers 계열
- retrieval: dense only
- generator/judge: 후속 단계
- 대규모 fine-tuning, reranker, hybrid retrieval은 현재 논문 범위 밖
