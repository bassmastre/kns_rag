# kns_rag — 프로젝트 컨텍스트

## 목적
한국원자력학회(KNS) 추계 학술대회 제출용 짧은 실험 논문(1~5p, domestic conference 수준).
주제: 원자력 규제 문서(STS) 대상 RAG에서 청킹 전략 비교.

## 핵심 연구 설계 (확정)
- 독립변수: **청킹 전략 하나만**. 임베딩·생성 LLM·판정 LLM·검색 방식은 전부 고정.
- 비교 4개 전략: naive 고정길이 / sliding window / semantic / 구조 인식형(제안 방법, condition 경계 기반).
- 핵심 주장: STS는 한 condition 안에서 AND/OR 논리 종속성으로 요건이 묶임. 의미 기반(semantic) 경계 탐지는 이 종속성을 끊음. 조항(condition) 경계로 자르는 구조 인식형이 semantic 포함 도메인 무관 방식보다 우수.
- 검색: dense-only, 단일 단계. 구조 인식형은 flat 청크에 section path만 prepend(2단계 검색 아님 — 청킹을 유일 변수로 유지).
- 평가: 검색 Hit@k·MRR(청크 ID 집합 기반 containment), 생성 LLM-as-judge(정확도·groundedness).
- QA: 약 60문항, 5종(추출형~22 / 조건-액션 매핑~22 / 정의~6 / 근거~6 / 답없음~6).
- Gold evidence: flat 청크 ID 집합(PDF 좌표·char offset 아님). 전략 독립적이어야 containment 매칭이 4개 전략 전반에 유효.

## 코퍼스
- Primary: NUREG-1431 Vol.1 (STS, Westinghouse). Vol.2(Bases)는 후속 단계.
- 인덱싱: 전체 문서. Gold QA 추출 대상: 3.4.1 / 3.4.10 / 3.4.13 / 3.4.16. 3.4(RCS) 전반은 distractor용.

## 데이터 스키마 (전처리 산출물, data/processed/)
- `sections.jsonl`: LCO 조항당 1레코드. `actions_text`(prose; naive/sliding/semantic 원본) + `condition_blocks`(구조 메타데이터; 구조 인식형이 참조) 공존.
- `struct_chunks.jsonl`: condition-action 단위로 평탄화한 청크(구조 인식형 전략의 실제 청크).
- ※ 파일명은 JSON 구조(중첩 vs 평탄) 기준. 실험 전략명과 무관. 구 이름: hierarchical.jsonl→sections.jsonl, flat.jsonl→struct_chunks.jsonl.

## 파싱 (pdfplumber)
- `extract_text()`는 다단(multi-column) ACTIONS 테이블에서 컬럼 섞임 → 사용 불가. `extract_words()` + x0 경계 컬럼 크로핑 사용.
- `extract_tables()`는 0개 검출(NUREG에 ruling line 없음).
- 컬럼 경계(경험적): CONDITION≈72/98, ACTION번호≈234, ACTION본문≈274, COMPLETION TIME≈425~443.
- AND/OR connector는 **원문 텍스트에서 추출**. label group 번호로 추론 금지(C.2·E.2가 group 바뀌어도 OR인 반례 존재).
- 헤더 제거(HEADER_MARGIN), 푸터는 어휘 앵커로 검출.

## 알려진 이슈 (오늘 작업 대상)
1. Condition G에 SURVEILLANCE REQUIREMENTS 테이블이 섞여 들어감. ACTIONS 테이블 하단 경계 미절단. → layout.py에 "SURVEILLANCE REQUIREMENTS" 앵커 컷 추가 필요.
2. 3.4.10/3.4.13/3.4.16 페이지 번호 미확정(config엔 3.4.15만 등록).
3. C/D/E의 optional:true가 실제 대괄호 조항인지 원문 대조 필요(오탐 시 gold 틀어짐).

## 코드 스타일
- 클래스 허용(기존 금지 원칙 해제). 단 과도한 추상화·조기 일반화 지양.
- 스코프 밖 모듈 미리 만들지 말 것. chunking/·span_index/ 스켈레톤은 폐기 설계(2단계 검색·좌표 gold) 반영 중 → 걷어낼 대상.
- 버그는 반복 수정으로 잡음.

## 응답 스타일
- 결론 먼저. 서론·격려 문구 없이 간결하게.
- 한국어로 논의. RAG 표준 용어(chunk, retrieval, Hit@k 등)는 영어 유지 가능. 비기술 용어의 문장 중간 영어 삽입·즉석 조어 금지.
- 확정 사항 vs 참고 논문 내용 엄격히 구분. 참고 논문 세부를 프로젝트 확정 결정으로 취급 금지.

## 환경
- HW: RTX 3060 12GB 1장. 대규모 파인튜닝 배제. API 사용 시 비용 언급.
- env: conda `kns_rag`, Python 3.11. 기간: 약 1개월.# kns_rag — 프로젝트 컨텍스트

## 목적
한국원자력학회(KNS) 추계 학술대회 제출용 짧은 실험 논문(1~5p, domestic conference 수준).
주제: 원자력 규제 문서(STS) 대상 RAG에서 청킹 전략 비교.

## 핵심 연구 설계 (확정)
- 독립변수: **청킹 전략 하나만**. 임베딩·생성 LLM·판정 LLM·검색 방식은 전부 고정.
- 비교 4개 전략: naive 고정길이 / sliding window / semantic / 구조 인식형(제안 방법, condition 경계 기반).
- 핵심 주장: STS는 한 condition 안에서 AND/OR 논리 종속성으로 요건이 묶임. 의미 기반(semantic) 경계 탐지는 이 종속성을 끊음. 조항(condition) 경계로 자르는 구조 인식형이 semantic 포함 도메인 무관 방식보다 우수.
- 검색: dense-only, 단일 단계. 구조 인식형은 flat 청크에 section path만 prepend(2단계 검색 아님 — 청킹을 유일 변수로 유지).
- 평가: 검색 Hit@k·MRR(청크 ID 집합 기반 containment), 생성 LLM-as-judge(정확도·groundedness).
- QA: 약 60문항, 5종(추출형~22 / 조건-액션 매핑~22 / 정의~6 / 근거~6 / 답없음~6).
- Gold evidence: flat 청크 ID 집합(PDF 좌표·char offset 아님). 전략 독립적이어야 containment 매칭이 4개 전략 전반에 유효.

## 코퍼스
- Primary: NUREG-1431 Vol.1 (STS, Westinghouse). Vol.2(Bases)는 후속 단계.
- 인덱싱: 전체 문서. Gold QA 추출 대상: 3.4.1 / 3.4.10 / 3.4.13 / 3.4.16. 3.4(RCS) 전반은 distractor용.

## 데이터 스키마 (전처리 산출물, data/processed/)
- `sections.jsonl`: LCO 조항당 1레코드. `actions_text`(prose; naive/sliding/semantic 원본) + `condition_blocks`(구조 메타데이터; 구조 인식형이 참조) 공존.
- `struct_chunks.jsonl`: condition-action 단위로 평탄화한 청크(구조 인식형 전략의 실제 청크).
- ※ 파일명은 JSON 구조(중첩 vs 평탄) 기준. 실험 전략명과 무관. 구 이름: hierarchical.jsonl→sections.jsonl, flat.jsonl→struct_chunks.jsonl.

## 파싱 (pdfplumber)
- `extract_text()`는 다단(multi-column) ACTIONS 테이블에서 컬럼 섞임 → 사용 불가. `extract_words()` + x0 경계 컬럼 크로핑 사용.
- `extract_tables()`는 0개 검출(NUREG에 ruling line 없음).
- 컬럼 경계(경험적): CONDITION≈72/98, ACTION번호≈234, ACTION본문≈274, COMPLETION TIME≈425~443.
- AND/OR connector는 **원문 텍스트에서 추출**. label group 번호로 추론 금지(C.2·E.2가 group 바뀌어도 OR인 반례 존재).
- 헤더 제거(HEADER_MARGIN), 푸터는 어휘 앵커로 검출.

## 알려진 이슈 (오늘 작업 대상)
1. Condition G에 SURVEILLANCE REQUIREMENTS 테이블이 섞여 들어감. ACTIONS 테이블 하단 경계 미절단. → layout.py에 "SURVEILLANCE REQUIREMENTS" 앵커 컷 추가 필요.
2. 3.4.10/3.4.13/3.4.16 페이지 번호 미확정(config엔 3.4.15만 등록).
3. C/D/E의 optional:true가 실제 대괄호 조항인지 원문 대조 필요(오탐 시 gold 틀어짐).

## 코드 스타일
- 클래스 허용(기존 금지 원칙 해제). 단 과도한 추상화·조기 일반화 지양.
- 스코프 밖 모듈 미리 만들지 말 것. chunking/·span_index/ 스켈레톤은 폐기 설계(2단계 검색·좌표 gold) 반영 중 → 걷어낼 대상.
- 버그는 반복 수정으로 잡음.

## 응답 스타일
- 결론 먼저. 서론·격려 문구 없이 간결하게.
- 한국어로 논의. RAG 표준 용어(chunk, retrieval, Hit@k 등)는 영어 유지 가능. 비기술 용어의 문장 중간 영어 삽입·즉석 조어 금지.
- 확정 사항 vs 참고 논문 내용 엄격히 구분. 참고 논문 세부를 프로젝트 확정 결정으로 취급 금지.

## 환경
- HW: RTX 3060 12GB 1장. 대규모 파인튜닝 배제. API 사용 시 비용 언급.
- env: conda `kns_rag`, Python 3.11. 기간: 약 1개월.