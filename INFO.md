# INFO — 연구 방향·가설·설계 결정

이 문서는 **왜** 이 실험을 이렇게 설계했는지를 정리한다. (무엇을·어떻게
실행하는지는 [`README.md`](README.md).) 협업 중 방향이 흔들릴 때 이 문서가
기준점이다. 여기 적힌 것이 확정 사항이고, 참고 논문의 세부는 확정 사항이 아니다.

---

## 1. 연구 질문

> 청킹 전략이, **식별자가 조밀하고 구조가 강한** 원자력 규제 문서 위에서
> RAG 검색·생성 성능에 어떤 영향을 주는가?

일반 도메인에서 semantic 청킹은 흔히 강한 baseline으로 취급된다. 이 연구는
그 통념이 STS 같은 규제 문서에서 **깨진다**는 것을 보이려 한다.

## 2. 핵심 주장 (thesis)

**"더 정교한 청킹"이 아니라 "도메인 구조를 아는 청킹"이 이긴다.**

STS의 ACTIONS 표는 하나의 CONDITION 아래 여러 Required Action이 **AND/OR
논리로 묶여** 있다. 이 논리적 묶음이 검색 단위의 최소 의미 단위다. 조각을
따로 회수하면 조치의 충분·필요 관계가 무너진다.

- 정방향: "이 조건에서 뭘 해야 하나" → 조건 아래 모든 액션(+AND/OR)이 한
  청크에 있어야 완전한 답.
- 역방향: "이 조치는 어떤 조건이 트리거하나" → 조건문과 액션이 같은 청크에
  있어야 답 가능.

**semantic 청킹의 실패 메커니즘**: 임베딩 유사도는 "grab sample 채취"와
"모니터 복구"를 의미가 다르다고 보고 경계를 그을 수 있다. 그러나 규제
논리상 이 둘은 한 조건의 필수 조치 쌍(AND) 또는 대안 쌍(OR)이다. semantic이
이를 가르면 부분 회수 → groundedness 저하로 이어진다.

**hierarchical(제안)**: CONDITION 경계를 청크 경계로 삼아 이 논리를 통째로
보존한다. 도메인 무관 기법은 구조를 모르므로 이 보존을 보장하지 못한다.

## 3. 가설 (검증 대상)

- **H1 (1차)**: 도메인 무관 semantic 청킹은 이 도메인에서 hierarchical보다
  검색 성능(Hit@k·MRR)이 낮다.
- **H2**: 특히 condition–action mapping 유형 QA에서 격차가 크다
  (AND/OR 논리가 직접 관여하므로).
- **보조 관찰**: naive/sliding은 구조를 모르지만 큰 청크를 우연히 만들어
  일부 조건을 통째로 담을 수 있다 → semantic만큼 명확히 지지 않을 수 있다.
  이 경우 "semantic의 정교함이 오히려 독"이라는 서사가 강화된다.

### 반증 조건 (미리 명시)

정직한 실험을 위해 가설이 **틀렸다고 볼 조건**을 먼저 못박는다.

- dense retriever가 쪼개진 액션 청크(E.1·E.2 등)를 top-k에 **모두** 끌어오면
  H1은 성립하지 않는다. 이 경우 결과를 그대로 보고한다.
- hierarchical의 우위가 **평균 청크 크기 차이만으로 설명**되면(큰 청크가
  containment Hit@k에 자명하게 유리) 구조 효과는 미입증으로 본다.
  → 그래서 모든 검색 결과에 **평균 청크 크기·개수를 병기**한다.

## 4. 실험 설계 원칙

### 4.1 단일 독립변수

청킹 전략만 변수. 임베딩·generator·judge·retrieval 방식·QA 언어는 전부 고정.
이것이 이 논문의 내적 타당성의 핵심이다. 어떤 확장 유혹(retriever 비교,
2단계 검색, reranker 등)도 이 원칙을 깨면 배제한다.

### 4.2 4개 전략

naive_fixed_length / sliding_window / semantic / hierarchical(제안).
앞 셋은 도메인 무관 baseline, 넷째만 STS 구조를 활용.

- 비교군(앞 셋)의 입력: `actions_text`(연속 줄글)
- 제안(hierarchical)의 입력: `condition_blocks`(구조) → condition–action 청크
- hierarchical은 **dense 단일 단계**. flat 청크에 section path만 prepend.
  summary node·2단계 검색은 **쓰지 않는다**(단일 변수 원칙 위반이므로).

### 4.3 semantic 파라미터 고정

breakpoint percentile은 **사전에 하나로 고정**(문헌 표준값, 예: p95).
결과를 보고 우리에게 유리한 값을 고르는 것은 금지. (파일럿에서 임계값을
바꾸면 최소 한 조건은 항상 쪼개진다는 점은 확인됨 — INFO §7 참조.)

### 4.4 평가

- **Retrieval**: Hit@k(k=1,3,5), MRR. gold = flat chunk id 집합에 대한
  **containment**. 보조로 IoU. 청크 크기·개수 병기.
- **Generation**: LLM-as-judge — accuracy(정답 일치), groundedness(회수된
  청크로 뒷받침되는가).
- gold를 chunk id 집합으로 잡는 이유: 전략마다 청크 경계가 달라 PDF 좌표로는
  전략 간 비교가 불가능. chunk id containment는 전략 독립적으로 유효.

### 4.5 QA 데이터셋

- 규모: 60~62문항. 통계 검정은 aggregate ~44문항 기준.
- 5종(자연 분포 유지 — 특정 방법에 유리한 유형으로 편향 금지):
  extractive ~22 / condition_action_mapping ~22 / definition ~6 /
  rationale(Vol.2) ~6 / unanswerable ~6.
- **type별 breakdown 보고**로 AND/OR 우위가 aggregate를 부풀리지 않고도
  드러나게 한다.
- 생성은 LLM, **gold 청크 매핑·unanswerable 판정은 사람 검증**.
  (생성 모델이 정답 청크를 보고 질문을 만들면 표현이 겹쳐 검색이 쉬워지는
  편향이 생기므로, 매핑 검증은 생략 불가.)
- QA 언어: **영어**(원문 매칭·containment 태깅에 유리). 논문 본문은 한국어.

## 5. 코퍼스 전략

- Primary: NUREG-1431 Vol.1 (STS, Westinghouse). Vol.2(Bases)는 rationale
  QA용으로 후순위.
- 인덱싱은 3.4(RCS) **전체** — distractor 확보. "LCO 3.4.1 vs 3.4.2" 같은
  식별자 혼동을 검색이 견디는지 보려면 유사 조항이 인덱스에 많아야 한다.
- QA gold 추출 대상은 그 부분집합. 선택 섹션이 condition 구조 복잡도
  (단순 조건 ~ AND/OR 중첩 조건)의 스펙트럼을 커버하도록 한다(리뷰 방어).

## 6. 도메인 문서 특성 (파싱이 어려운 이유)

- ACTIONS 표는 **ruling line이 없는** 다단 레이아웃(whitespace 정렬).
  → `extract_tables()`는 0개 검출, `extract_text()`는 컬럼을 섞어버림.
  → `extract_words()` + x0 좌표 밴드 파싱으로 컬럼을 직접 가른다.
- AND/OR connector는 **원문 텍스트가 authoritative**. 라벨 group 번호로
  추론하면 틀린다(C.2·E.2가 group이 바뀌어도 OR인 반례 존재).
- optional 조치는 원문 대괄호 `[ ]`로 표기(플랜별 선택 조항). 파서가 이를
  잡되, 실제 대괄호와 파싱 아티팩트를 구분해야 한다(원문 대조로 검증).
- 한 페이지에 ACTIONS 표와 SURVEILLANCE REQUIREMENTS 표가 이어지면 경계가
  안 끊겨 오염될 수 있다(3.4.15 Condition G에서 확인·수정됨).

## 7. 파일럿 결과 (기록)

`scripts/pilot_semantic_vs_structure.py` — 3.4.15 하나로 "semantic이
condition 내부 AND/OR 짝을 경계로 가르는가"만 확인한 **1회성 sanity check**.

- 관찰: breakpoint를 p75/p90/p95로 바꿔도 최소 한 조건(D)은 **항상** 쪼개짐.
  p95에서 E가 우연히 안 갈린 건 문서가 거의 안 잘린 상태(청크 3개)였기 때문.
- **한계(중요)**: 이 파일럿은 청크 **경계**만 봤고 **검색 지표를 재지
  않았다**. 즉 "경계가 쪼개진다"(가설의 전제)를 확인했을 뿐, "검색이
  나빠진다"(H1 자체)를 입증하지 못한다. 또한 hierarchical 쪽 100% 보존은
  정의상 자명(경계 = condition)하므로 발견이 아니다.
- **위치**: 이 결과는 논문의 Results가 아니라 **방법론/논의의 메커니즘
  삽화**로만 쓴다. H1은 실제 retrieval 실험으로 검증해야 한다.

## 8. 스코프 밖 (명시적 배제)

혼동·확장을 막기 위해 **하지 않을 것**을 못박는다.

- retriever 비교(lexical/hybrid/dense)는 **별도 프로젝트**. 이 논문에 통합 금지.
- 2단계 검색·summary node·reranker — 단일 변수 원칙 위반.
- 대규모 파인튜닝 — HW 제약(RTX 3060 12GB).
- Vol.2 Bases 전면 활용 — rationale QA 소수만, 후순위.
- 파일럿 결과를 H1 입증으로 사용 — §7 참조.

## 9. 참고 문헌 맥락

- **Byun & Kim (KNS 2026 봄, KAERI)**: 원자력 RAG 파이프라인 논문이나
  청킹에 초점 없음 → 본 연구의 gap을 설정(직접 비판 아님).
- **MultiDocFusion (EMNLP 2025)**: hierarchical 전략의 개념적 영감.
  단 NUREG-1431은 born-digital의 명시적 구조를 가지므로 그 논문의 무거운
  vision/fine-tuning 스택은 불필요.

> 위 논문의 구체적 선택(모델·하이퍼파라미터 등)은 **참고일 뿐 본 프로젝트의
> 확정 사항이 아니다**. 확정 사항은 이 문서 §1–§8이다.

## 10. 논문 구조 (목표)

서론 → 관련연구 → 방법론(도메인 특성·4전략·평가) → 실험(코퍼스·QA·설정)
→ 결과(type별 breakdown·청크 크기 병기) → 결론. 분량 1~5p, 명확한 결론 하나.