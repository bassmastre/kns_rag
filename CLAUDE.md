# kns_rag — 프로젝트 컨텍스트

## 목적
한국원자력학회(KNS) 추계 학술대회 제출용 짧은 실험 논문(1~5p, domestic conference 수준).
주제: 원자력 규제 문서(NUREG-1431 STS) 대상 RAG에서 chunking 전략 비교. 논문은 영어.

## 핵심 연구 설계 (확정)
- 독립변수: **chunking 전략 하나만**. embedding·retrieval 방식은 전부 고정.
- 비교 5개 전략:
  - domain-agnostic baseline: `naive_fixed_length` / `sliding_window` / `semantic`
  - structure-aware: `action_logic` / `condition_aware`
- 핵심 주장: STS는 한 CONDITION 안에서 Required Action이 AND/OR 논리로 묶인다.
  domain-agnostic 경계 탐지는 이 종속성을 끊고, structure-aware는 보존한다.
  retrieval 지표로 측정 가능.
- retrieval: dense-only, 단일 단계. structure-aware도 flat chunk에 section path만
  prepend (2단계 검색 아님 — chunking을 유일 변수로 유지).
- **평가: retrieval-only 확정.** downstream generation + LLM-judge는 main 실험에서
  제외. 토대 안정화 + 시간 여유 시에만 Experiment 2 (stretch goal)로 검토.
- QA 5종: `extractive` / `condition_action_mapping` / `multi_condition` /
  `completion_time` / `definition`. 목표 ~60문항, LLM 생성 + **사람 검증 필수**.
  - `unanswerable`은 **제외 확정** (gold 공집합 → retrieval 지표에서 구조적으로
    빠지고, chunking-invariant해서 실험 축과 어긋남).
  - Vol.2 rationale 유형은 코퍼스가 보류이므로 없음.
- **gold 채점은 keyword containment가 main 기준.** chunk ID exact-match 아님.
  근거: 아래 "채점 기준" 절.
- gold는 **전략 독립**이어야 한다. 특정 전략의 chunk 경계에 맞춰 QA를 만들면 편향.

## 코퍼스
- Primary: NUREG-1431 Vol.1 (STS, Westinghouse). **Vol.2(Bases)는 보류 확정.**
- 인덱싱 범위와 QA 추출 범위는 `config.yaml`의 `sections` / `qa_sections` 기준.
  (두 값의 현재 등록 내용이 의도와 맞는지는 미확정 — 아래 "열린 결정" 참고.)
- SURVEILLANCE REQUIREMENTS 섹션은 코퍼스에서 **제외 확정**.
  `layout.find_sr_heading_top()`이 ACTIONS 표/narrative의 하단 컷으로 적용한다.
- 대괄호 표기(`[Two] [required]` → `Two required`)는 제거된다. NUREG-1431의
  대괄호는 발전소별 입력값 표기. 5전략 공통이라 교란 변수 아님.
  **단, 논문 코퍼스 기술에 "대괄호 표기 제거" 명시할 것.**

## 데이터 스키마 (전처리 산출물, data/processed/)
`corpus.build_records(section)`이 3개를 **한 튜플로 동시 생성**한다.
- `raw.jsonl` — LCO 조항당 1레코드. flattened `raw_text`.
  → `naive_fixed_length` / `sliding_window` / `semantic`의 입력.
- `hierarchical_source.jsonl` — action 단위 레코드. → `action_logic`의 입력.
- `condition_chunks.jsonl` — condition block 단위 레코드. → `condition_aware`의 입력.

※ 구 이름(`sections.jsonl`, `struct_chunks.jsonl`, `flat.jsonl`, `hierarchical.jsonl`)은
전부 폐기. 코드에 존재하지 않는다.

### unit ID 네임스페이스
`hierarchical_source.jsonl`의 id는 **action 단위 + LCO 통짜 1개**다.
3.4.5 예: `3.4.5/A.1`, `/B.1`, `/C.1`, `/C.2`, `/D.1`, `/D.2`, `/D.3`, `/LCO` (8개).
- 조건 단독(`/A`, `/D`)도, LCO 하위 절(`/LCO.a`, `/LCO.rcp_note`)도 **없다.**
- `3.4.5/LCO`는 LCO 본문 + RCP note를 통째로 담는 단일 unit.
- **존재하지 않는 ID를 gold에 쓰면 전 전략 null hit — silent failure mode.
  실제로 발생한 적 있다.** gold ID는 실제 네임스페이스와 대조할 것.

## 채점 기준 (keyword containment를 main으로 쓰는 이유)
`metadata.evidence_ids` 부여 방식이 **전략별로 다르다**:
- `action_logic`(`strategies.py:465`) / `condition_aware`(`:484`) → `hierarchical_source`의
  실제 id. **정확.**
- `naive_fixed_length`(`:142`) / `sliding_window`(`:176`) / `semantic`(`:444`) →
  `_evidence_ids_from_text()`(`:59-72`)의 **정규식 추측.**

정규식 경로의 실측 오류 3종:
- **LCO 오탐**: 다른 LCO를 상호참조하는 문장(`"...to meet SDM of LCO 3.1.1."`)에
  `"LCO"` 세 글자가 있다는 이유로 `{lco}/LCO` 크레딧(`:70`).
- **라벨-only 통과**: 행동 텍스트가 잘려도 라벨만 남으면 커버로 인정 →
  **fragmentation 손상이 지표에 안 잡힌다. thesis가 재려는 현상 자체를 못 본다.**
- **텍스트-only 탈락**: 행동 전문을 담아도 라벨이 없으면 미인정.

→ **구조 전략은 정확한 ID, baseline은 정규식 추측. 서로 다른 자로 잰 값이라
ID 경로는 공정 비교 불가.** keyword containment는 5전략 전부에 동일 기준
("chunk body에 판별 구절이 있나")을 적용한다. **논문 Methodology에 명시할 것.**
선행 사례: CoFE-RAG.

`evaluation.py` 시맨틱:
- `gold_keywords`의 **바깥 리스트 = gold unit 하나당 그룹 하나** (`gold_units`와 1:1).
- **그룹 내부는 AND**(`result_hits_keyword_group`, `:71`), **그룹 간은 OR**(`:61`).
- 지표: `hit@k`/`mrr`(하나라도 맞으면 hit, 너그러움), `coverage@k`(gold 그룹 커버
  비율, 부분 점수), `all_gold@k`(전부 커버해야 1, 엄격), `all_gold_rr`.
  ※ 내부 키는 `gold_coverage@k`, CSV/콘솔 표면형은 `coverage@k`.

## 파싱 (pdfplumber)
- `extract_text()`는 다단 ACTIONS 표에서 컬럼 섞임 → **사용 불가.**
  `extract_words()` + x0 경계 컬럼 분리 사용.
- `extract_tables()`는 0개 검출 (NUREG에 ruling line 없음).
- 컬럼 경계는 **`config.yaml`의 `layout.defaults`가 단일 출처** (`col1_max`,
  `col2_max`, `header_margin`). 코드에 하드코딩 금지. LCO별 예외는 `layout.overrides`.
- 완료시간 바인딩: band 기반 top-y containment. 본문 뒤에 세로로 쌓인 경우도
  정상 동작 (3.4.3 Condition B: B.1→6 hours, B.2→36 hours로 검증 완료).
- **AND/OR connector는 원문 텍스트가 유일한 authoritative source.**
  label 번호로 유추 **금지**. 반례: 3.4.5에서 C.1→C.2는 OR, D.1→D.2→D.3은 AND.
  번호 증가 패턴이 같은데 connector가 반대. (`corpus.py:100` 주석 참조.)
- connector 주입(`corpus.condition_logic`, `_condition_inter_clauses`)은 ① 가까운
  boundary 먼저 ② 자기 group을 절 앞에. **D.3의 절 순서가 D.1/D.2와 다른 것은
  거리 정렬의 정상 결과이지 버그가 아니다. 수정 불필요.**
- 아래첨자: `join_words()`가 visual row 클러스터링 후 `_is_subscript()` 좌표 판정으로
  앞 토큰에 결합한다. **표면형은 `T_avg`** (원문 렌더는 `Tavg`).
  → **keyword containment는 문자열 매칭이다. gold_keywords는 이 표면형 기준으로
  작성해야 한다.**

## 코드 스타일
- 클래스 허용. 단 과도한 추상화·조기 일반화 지양.
- 스코프 밖 모듈 미리 만들지 말 것.
- **금지**: `pipeline.py` / `generation.py` / `validation.py` / `corpus` 서브패키지 /
  `DenseIndex` 클래스 생성. 산출물 스키마·파일명 변경.
- 산출물 회귀 검증 기준은 "돌아감"이 아니라 **"산출물이 같음"**:
  `raw.jsonl` / `hierarchical_source.jsonl` / `condition_chunks.jsonl` **3개 전부**
  레코드 단위 diff 0 + index `embeddings.npy` shape · `meta.json` · `chunks.jsonl` 동일.
  (셋은 같은 `build_records(s)` 튜플에서 나오지만 따로 깨질 수 있다.
  `condition_chunks`는 제안 전략 입력이라 회귀가 가장 치명적.)

## 응답 스타일
- 결론 먼저. 서론·격려 문구 없이 간결하게.
- 한국어로 논의. RAG/IR 표준 용어(chunk, retrieval, Hit@k, MRR)는 영어 유지.
  비기술 용어의 문장 중간 영어 삽입·즉석 조어 금지. 새 용어는 사용 전 합의.
- 확정 사항 vs 참고 논문 내용을 엄격히 구분. 참고 논문 세부를 이 프로젝트의
  확정 결정으로 취급 금지. 읽거나 논의한 것 ≠ 결정하거나 구현한 것.

## 검증 규율
- **파일이나 코드를 읽어야 판단되는 것은, 읽기 전에 제안하지 말 것.**
  파일을 요청하고 **멈춘다.**
- **"확인 필요"는 정지 신호이지 추측을 계속 밀어도 된다는 면허가 아니다.**
- **작업 항목을 만들기 전에 "이미 했는지"를 먼저 묻는다.**
- 주장할 때 근거를 구분한다: **코드/데이터로 확인함 / 추론함 / 가정함.**

## 열린 결정 (미확정 — 임의로 닫지 말 것)
- **main 지표**: `hit@k` vs `all_gold@k`/`coverage@k`. 어느 전략이 이기는지가 바뀐다.
- **`action_logic` vs `condition_aware`**: 어느 쪽이 제안 전략인지 미결정.
- **`completion_time` 유형 처리**: `sliding_window`가 우세한 thesis 반증 방향.
- **QA 유형별 문항 수 배분**: 총 ~60 외에 배분 미확정.
- **`sections` / `qa_sections`의 현재 등록 범위가 의도와 맞는지.**
- **embedding model / (Experiment 2로 갈 경우) judge LLM 최종 확정.**
- **그룹 내 AND 시맨틱이 의도된 것인지.** 파일럿 keyword 작성 시엔 OR로 상정했으나
  코드는 AND. 현재 파일럿 수치는 AND 기준으로 계산된 값.

## 환경
- HW: RTX 3060 12GB 1장. 대규모 파인튜닝 배제. API 사용 시 비용 명시.
- env: conda `kns_rag`, Python 3.11. `pip install -e .` (src-layout).