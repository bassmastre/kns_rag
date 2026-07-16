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


| 전략 | 역할 | 입력 소스 | 청킹 단위 |
|------|------|-----------|-----------|
| `naive_fixed_length` | domain-agnostic baseline | `raw.jsonl`의 flattened `raw_text` | 고정 길이 |
| `sliding_window` | domain-agnostic baseline | `raw.jsonl`의 flattened `raw_text` | 고정 길이 + overlap |
| `semantic` | domain-agnostic baseline | `raw.jsonl`의 flattened `raw_text` | embedding boundary |
| `action_logic` | structure-aware | `hierarchical_source.jsonl` | action 단위 (작음) |
| `condition_aware` | structure-aware | `condition_chunks.jsonl` | condition block 단위 (큼) |

> **`action_logic`과 `condition_aware` 중 어느 쪽이 제안 전략인지는 미확정.**
> 둘 다 structure-aware이며, 본 세트 결과 전에는 결정하지 않는다.

> **chunk 크기 confound**: 두 structure-aware 전략은 청킹 단위가 달라 크기가 다르다.
> retrieval 지표를 보고할 때 **평균 chunk 크기와 chunk 개수를 반드시 병기**할 것.
> (큰 chunk일수록 관련 passage 포함 확률이 높아 IR 지표가 편향된다 — growing-window
> 계열의 지적.)

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

생성물은 전부 `config.yaml`의 `paths` 아래로만 쓰인다 — repo 루트/cwd에
직접 쓰는 코드는 없다. `outputs/` 하위 서브디렉터리 이름(indexes, retrieval,
generation, eval)은 config 키가 아니라 `src/kns_rag/config.py` 한 곳의
상수다 (바꿀 일이 생기면 그 파일만 수정).

```text
data/
  raw/        원본 PDF (입력)
  processed/  01 산출: raw / hierarchical_source / condition_chunks
  chunks/     02 산출: <strategy>.jsonl
  qa/         사람 검증 QA (qa.jsonl) — 04 이후의 입력 경계
outputs/
  indexes/    03 산출
  retrieval/  04 산출
  generation/ 05 산출
  eval/       06 산출
```

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

설치 (editable install — 스크립트가 `kns_rag` 패키지를 import한다):

```bash
pip install -e .                    # 파싱/코퍼스만 (01~02 non-semantic)
pip install -e ".[retrieval]"       # + numpy, sentence-transformers (semantic/03 이후)
```

파일명 번호가 파이프라인 순서다. 모든 스테이지는 **인자 없이 실행 가능**하며
(`--config`는 repo의 config.yaml, `--strategy`는 all이 기본), override 인자는 유지된다.

### 01~03: QA 없이 사람 개입 없이 순서 실행

```bash
python scripts/01_build_corpus.py
python scripts/02_build_chunks.py
python scripts/03_build_index.py
```

### 04~06: `data/qa/qa.jsonl`(사람 검증 QA)이 있어야 동작

qa.jsonl은 파이프라인이 만들어 주지 않는 **사람 검증 산출물**이다.
없으면 04부터는 정상적으로 실패한다 (코드 오류가 아니라 입력 부재).

```bash
python scripts/04_retrieve.py
python scripts/05_build_rag_inputs.py
python scripts/06_eval_retrieval.py
```

### 유틸 (번호 없음 = 선형 스테이지 아님)

```bash
python scripts/qa_smoke.py     # 스모크용 QA 자동 생성 (최종 평가에 사용 금지)
python scripts/run_pipeline.py --from-stage corpus --to-stage index
```

특정 전략만:

```bash
python scripts/02_build_chunks.py --strategy condition_aware
python scripts/03_build_index.py --strategy condition_aware
```

---

## QA 스키마

**채점은 `gold_keywords` containment가 main 기준이다.** chunk ID exact-match는
전략 간 공정 비교가 불가능하다(evidence_ids 부여 방식이 구조 전략은 정확한 id,
baseline은 정규식 추측 — `CLAUDE.md`의 "채점 기준" 절 참고).

```json
{"id":"q001","type":"condition_action_mapping","question":"...","gold_keywords":[["Restore required RCS loop to OPERABLE status"],["Be in MODE 3"]],"gold_evidence_ids":["3.4.5/C.1","3.4.5/C.2"],"answerable":true}
```

`gold_keywords` 시맨틱:
- **바깥 리스트 = gold unit 하나당 그룹 하나.** `gold_evidence_ids`와 1:1 대응.
- **그룹 내부는 AND** — 그룹의 모든 keyword가 같은 chunk body에 있어야 매칭.
- **그룹 간은 OR** — `hit@k`/`mrr` 계산용. `all_gold@k`/`coverage@k`는 그룹 커버
  개수를 센다.
- 매칭은 소문자화 + 공백 정규화 후 substring (`normalize_match_text`).
  **아래첨자 표면형은 `T_avg`다** (원문 렌더는 `Tavg`). keyword를 이 표면형으로 쓸 것.

keyword 선정: **판별력 있는 3~6 토큰 구절.** STS는 도메인 어휘가 반복되므로
짧은 n-gram은 false positive를 만든다. 반대로 문장 전체는 chunk 경계를 넘으면
매칭이 깨진다.

> 알려진 충돌: 3.4.5에서 `C.2`와 `D.1`의 원문 문구가 동일하다
> (`"Place the Rod Control System in a condition incapable of rod withdrawal"`).
> 전체 3.4로 확장 시 이런 충돌이 늘어난다. gold 작성 시 대조할 것.

`gold_evidence_ids`는 **참조·감사용으로만 유지한다** (채점 경로 아님).
허용 필드명: `gold_evidence_ids` / `evidence_ids` / `gold_ids` / `gold.evidence_ids`.

> **gold ID는 실제 unit ID 네임스페이스에 존재해야 한다.** 논리 단위 ID를
> 지어내면 전 전략 null hit — silent failure mode. 실제로 발생한 적 있다.

`type`이 `unanswerable`인 레코드는 retrieval metric에서 제외된다(`is_answerable`).
**단 `unanswerable` 유형은 본 실험에서 제외 확정이므로 데이터셋에 존재하지 않는다.**
이 코드 경로는 방어용으로만 남아 있다.

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
  config.py        config 로딩 + 경로 계산의 단일 지점 (ExperimentConfig)
  io.py            JSONL/JSON IO + chunk_body/evidence_ids 접근자
  text.py          정제·정규화 + 라벨/커넥터 정규식
  layout.py        좌표 유틸 + 페이지 영역 분할
  parse.py         PDF page → raw dict
  corpus.py        section merge → processed corpus
  chunking/        청킹 전략 구현 (strategies.py)
  embeddings.py    SentenceTransformer embedding utility
  retrieval.py     dense index / retrieval utility
  evaluation.py    Hit@k / MRR evaluation utility
scripts/
  01_build_corpus.py
  02_build_chunks.py
  03_build_index.py
  04_retrieve.py                   # 여기부터 data/qa/qa.jsonl 필요
  05_build_rag_inputs.py
  06_eval_retrieval.py
  run_pipeline.py                  # 스테이지 러너
  qa_smoke.py                      # QA 준비 유틸 (선형 스테이지 아님)
  pilot_semantic_vs_structure.py   # 1회성 파일럿
pyproject.toml
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
