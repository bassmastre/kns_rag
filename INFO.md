# INFO — 연구 방향·가설·설계 결정

이 문서는 **왜** 이 실험을 이렇게 설계했는지를 정리한다. 무엇을 어떻게
실행하는지는 [`README.md`](README.md)를 기준으로 한다. 협업 중 방향이 흔들릴
때 이 문서가 기준점이다.

---

## 1. 연구 질문

> 구조가 강한 원자력 규제 문서(STS)에서 RAG는 어떤 유형의 질문에서 실패하며,
> 그 실패 중 어느 부분이 청킹 전략으로 완화되는가?

초기 질문은 "hierarchical chunking이 semantic chunking보다 우수한가"였지만,
현재 프레이밍은 더 좁고 정직하다. 핵심은 **RAG 실패 유형 진단 + 청킹 전략별
유효 범위 분석**이다.

---

## 2. 핵심 주장

**"더 정교한 청킹"이 아니라 "도메인 구조를 보존하는 청킹"이 유리하다.**

NUREG-1431 STS의 ACTIONS 표는 하나의 Condition 아래 Required Action과
Completion Time이 AND/OR 논리로 연결된다. 이 구조가 무너지면 retrieval은 관련
텍스트 일부를 회수해도 답변이 불완전해질 수 있다.

예상되는 실패 메커니즘:

- fixed-length: condition/action/CT가 물리적으로 잘림
- sliding-window: overlap으로 recall은 보완되지만 중복과 경계 노이즈 증가
- semantic: embedding coherence 기준으로 경계를 고르지만 STS의 logical boundary를
  보장하지 않음
- structure-aware: Condition–Required Action–Completion Time–connector를 한 단위로
  보존

---

## 3. 검증 가설

### H1 — retrieval

`condition_aware`는 condition-action mapping 유형에서 raw-text baseline보다 높은
Hit@k/MRR을 보일 가능성이 높다.

### H2 — failure mode

청킹 효과는 모든 실패를 해결하지 않는다. 특히 다음 실패는 청킹만으로 해결되지
않을 수 있다.

- 질문 자체가 문서에 없음(unanswerable)
- answer synthesis 단계의 수치/조건 오독
- Vol.1 specification만으로 rationale을 요구하는 질문
- retriever가 관련 chunk를 찾았지만 generator가 connector 논리를 잘못 해석하는 경우

### H3 — semantic chunking의 한계

semantic chunking은 도메인 무관 의미 경계를 고르므로 문장상 자연스러운 경계와
규제 논리상 필요한 경계가 어긋날 수 있다.

---

## 4. 전략 정의

본 실험의 구현상 전략은 5개다. 단, 논문에서의 주 비교축은 **raw-text baseline 3개
대 condition-aware structure chunking**이다. `action_logic`은 주 전략이 아니라 구조
단위 ablation이다.

| 전략 | 분류 | 설명 | 입력 |
|------|------|------|------|
| `naive_fixed_length` | baseline | 고정 길이 분할 | `raw.jsonl` |
| `sliding_window` | baseline | 고정 길이 + overlap | `raw.jsonl` |
| `semantic` | baseline | embedding 기반 word-boundary coherence | `raw.jsonl` |
| `action_logic` | ablation | Required Action 단위 구조 청크 | `hierarchical_source.jsonl` |
| `condition_aware` | proposed | Condition 단위 구조 청크 | `condition_chunks.jsonl` |

`condition_aware`는 Condition 하나에 해당 Condition text, Required Actions,
Completion Times, AND/OR connector를 함께 넣는다. 현재 논문의 제안 전략은 이것이다.

`action_logic`은 개별 Required Action을 하나의 chunk로 만들고 주변 connector 정보를
metadata/body에 포함한다. exact action retrieval에는 유리할 수 있으나, condition-level
질문에서는 여러 action chunk를 동시에 회수해야 하므로 주 전략으로 삼지 않는다.

---

## 5. 단일 변수 원칙

고정할 것:

- embedding model
- retrieval method: dense only
- query language: English
- indexed text field: `content.body`
- evaluation metric: Hit@k, MRR

변수:

- chunking strategy only

배제할 것:

- lexical/hybrid retriever 비교
- reranker
- 2-stage retrieval
- summary node
- model fine-tuning

이유: 위 요소를 넣으면 결과가 chunking 효과인지 retriever/model 효과인지 분리하기 어렵다.

---

## 6. semantic chunking 구현 해석

현재 semantic 전략은 자유롭게 임의 길이로 자르는 방식이 아니다. 사전에 정한 size
budget 안에서 word-boundary candidate를 만들고, 좌우 context embedding coherence가 낮은
경계를 선택한다.

고정 파라미터:

- `min_chars`
- `target_chars`
- `max_chars`
- `context_window_words`
- `boundary_step_words`
- embedding model

따라서 논문에는 다음처럼 기술하는 것이 안전하다.

> Semantic chunking selected embedding-based word boundaries under a predefined
> chunk-size budget, without using the regulatory Condition–Action structure.

semantic 결과가 사람이 보기엔 condition 문장을 중간에서 자르더라도, 알고리즘 결과라면
수동 보정하지 않는다. 그것이 baseline의 관찰 결과다.

---

## 7. 코퍼스와 산출물

Primary corpus:

- NUREG-1431 Vol.1, RCS 3.4.x 계열
- Vol.2 Bases는 rationale QA용 후속 단계

현재 processed corpus:

| 파일 | 의미 | downstream |
|------|------|------------|
| `raw.jsonl` | LCO section-level flattened text | fixed/sliding/semantic |
| `hierarchical_source.jsonl` | action-level structured source | action_logic / gold evidence basis |
| `condition_chunks.jsonl` | condition-level structured source | condition_aware |

주의: raw-text baseline과 structure-aware 전략은 같은 PDF extraction root에서 출발하지만,
입력 표현은 다르다. 따라서 논문 표현은 다음처럼 잡는다.

> We compare common raw-text chunking baselines with a structure-aware hierarchical
> chunk construction strategy.

피해야 할 표현:

> identical preprocessing with only split boundary changed

이 표현은 부정확하다.

---

## 8. Retrieval 평가 설계

Retrieval metric:

- Hit@1 / Hit@3 / Hit@5
- MRR
- type별 breakdown
- chunk count / average chunk length 병기

Gold evidence:

- 기본 단위는 `hierarchical_source.jsonl`의 action-level evidence id
- 예: `3.4.5/C.1`, `3.4.5/C.2`, `3.4.5/LCO`

`metadata.evidence_ids`는 retrieval input이 아니다. embedding/indexing 대상은 항상
`content.body`뿐이다. 따라서 evidence id 문제는 검색 노이즈 문제가 아니라 evaluation
label mapping 문제다.

---

## 9. QA 데이터셋 원칙

목표 규모:

- 약 60문항
- extractive: 약 22
- condition-action mapping: 약 22
- definition: 약 6
- rationale: 약 6
- unanswerable: 약 6

통계적 비교의 중심은 extractive + condition-action mapping이다. rationale과 unanswerable은
RAG 실패 유형 분석 및 논의용 성격이 강하다.

QA 작성 원칙:

- 질문은 영어
- gold evidence mapping은 사람 검증
- 자동 생성 QA는 pipeline smoke test에는 사용할 수 있으나 최종 실험 결과에는 사용하지 않음
- final dataset에서는 특정 전략에 유리한 표현 반복을 피함

---

## 10. Generation / Judge 계획

현재 구현은 retrieval evaluation과 RAG prompt 입력 생성까지다. 실제 generator 호출과
LLM-as-judge는 후속 단계다.

Generation 평가 항목:

- answer correctness
- groundedness
- hallucination / unsupported answer
- connector logic preservation

Judge는 같은 prompt/template을 모든 전략에 적용한다. retrieval context만 전략별로 달라진다.

---

## 11. 실패 taxonomy

논문 결과는 단순 평균 점수만 보고하지 않는다. 다음 failure type별로 분석한다.

| Failure type | chunking으로 완화 가능성 |
|--------------|--------------------------|
| condition/action split | 높음 |
| completion time/action mismatch | 높음, 구조 보존 시 |
| AND/OR connector omission | 중간~높음 |
| cross-row dependency | 중간 |
| number/operator mismatch | 낮음, generation 문제 가능 |
| rationale unavailable in Vol.1 | 낮음 |
| unanswerable question | 낮음 |

핵심 결론은 "condition_aware가 모든 것을 해결한다"가 아니라, **청킹으로 해결 가능한
실패와 그렇지 않은 실패를 분리하는 것**이다.

---

## 12. 현재 pipeline 상태

구현 완료:

```text
corpus → chunks → index → retrieve → rag_inputs → eval
```

QA 없이 실행 가능한 단계:

```text
corpus → chunks → index
```

QA가 필요한 단계:

```text
retrieve → rag_inputs → eval
```

따라서 `data/qa/qa.jsonl`이 없으면 `04_retrieve.py`와 `run_pipeline.py --to-stage eval`은
정상적으로 실패한다. 이것은 코드 오류가 아니라 입력 데이터 부재다.

스크립트 파일명의 번호(01~06)가 위 스테이지 순서를 그대로 인코딩한다.
번호 없는 스크립트(`qa_smoke.py`, `pilot_semantic_vs_structure.py`,
`run_pipeline.py`)는 선형 스테이지가 아니다.

---

## 13. 스코프 밖

- retriever 비교 실험
- BM25/hybrid/reranker 추가
- Graph RAG
- fine-tuning
- Vol.2 Bases 전면 통합
- QA 자동 생성 결과를 최종 metric으로 사용
- semantic boundary를 사람이 수동 보정

---

## 14. 논문 구조 목표

1. Introduction
2. Related Work
3. Corpus and Document Structure
4. Chunking Strategies
5. Experimental Setup
6. Retrieval Results
7. Failure Analysis
8. Discussion and Limitations
9. Conclusion

분량이 1~5p이면 Results와 Failure Analysis를 합치고, Generation 평가는 retrieval 결과가
정리된 뒤 선택적으로 축소한다.
