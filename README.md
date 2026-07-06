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

## 독립변수 = 청킹 전략 (유일)

| 전략 | 설명 | 입력 소스 |
|------|------|-----------|
| naive_fixed_length | 고정 길이 | 줄글(actions_text) |
| sliding_window | 겹치는 윈도우 | 줄글(actions_text) |
| semantic | 문장 임베딩 유사도로 경계 탐지 | 줄글(actions_text) |
| hierarchical (제안) | STS 구조(condition 단위)를 경계로 분할 | 구조(condition_blocks) |

임베딩·generator·judge·retrieval·언어는 전부 고정. 청킹만 변수.

---

## 현재 상태 (2026-07 기준)

구현이 완료된 단계와 미완 단계를 명확히 구분한다.

**완료**
- PDF 파싱 파이프라인(`parse.py` + `layout.py` + `text.py` + `corpus.py`)
- 3.4.15 파싱 검증: connector(AND/OR) 원문 추출, 대괄호 optional 액션,
  NOTE, LCO statement, Condition G의 SURVEILLANCE 오염 절단까지 정합 확인
- 코퍼스 산출물 2종 생성(`build_corpus.py`)

**진행 중 / 미완**
- 3.4 전체(3.4.1~3.4.20) 파싱 확장 — config에 페이지 범위 등록됨, 검증 미완
- 출력 파일명 리네임(아래 "네이밍" 참조) — **미적용**
- QA 데이터셋 생성 — 미착수
- 4개 전략 청킹 / retrieval / eval 구현 — 미착수
  (`chunking/`, `span_index/`는 **폐기 예정 스켈레톤**, 아래 참조)
- 임베딩·generator·judge 모델 확정 — config에 placeholder(`null`)

---

## 코퍼스

- **문서**: NUREG-1431 Vol.1 (STS, Westinghouse). Vol.2(Bases)는 후순위.
- **인덱싱 범위**: 3.4(RCS) 전체 — distractor 확보용.
- **QA 추출 대상**: `config.yaml`의 `qa_sections`(현재 3.4.1~3.4.20 등록).
  실제 gold 작성 대상 축소 여부는 INFO.md의 QA 설계 참조.

### 산출물 (`data/processed/`)

| 파일 | 단위 | 용도 |
|------|------|------|
| `sections.jsonl` | LCO 조항당 1레코드 | `actions_text`(줄글) + `condition_blocks`(구조) 공존 |
| `struct_chunks.jsonl` | condition–action 단위 flat 레코드 | hierarchical 전략 청크 / gold 정의 기준 |

- `sections.jsonl`의 `actions_text` → naive/sliding/semantic 입력
- `sections.jsonl`의 `condition_blocks` → hierarchical 입력
- `struct_chunks.jsonl`의 청크 ID 집합 → **gold evidence 정의 기준**
  (원문 좌표·char offset 아님)

### 네이밍 (리네임 예정, 미적용)

파일명이 실험 전략명(`hierarchical`)과 겹쳐 혼동을 유발하므로 아래로 변경 예정:

- `sections.jsonl` → `raw.jsonl` (비교군 청킹 소스)
- `struct_chunks.jsonl` → `hierarchical_source.jsonl` (구조 인식형 청킹 소스)

> `_source` 접미사로 "전략 자체가 아니라 그 전략의 입력"임을 명시.
> **아직 코드에 반영 안 됨** — 적용 시 `build_corpus.py` 출력부와
> 다운스트림 참조를 함께 갱신할 것.

---

## 평가

- **Retrieval**: Hit@k(k=1,3,5), MRR — containment 기반(IoU 보조).
  평균 청크 크기·개수 병기(chunk size confound 관리).
- **Generation**: LLM-as-judge (accuracy, groundedness).
- **Gold evidence**: flat chunk id 집합(전략 독립).

---

## 코드 구조

```
src/kns_rag/
  text.py          정제·정규화 + 라벨/커넥터 정규식
  layout.py        좌표 유틸 + 페이지 영역 분할(narr/tbl)
  parse.py         PDF 페이지 → raw dict (컬럼 밴드 파싱)
  corpus.py        섹션 병합 → hierarchical 레코드 + flat 레코드
  preprocessing/   [폐기 예정] 별도 PDF 분할 스크립트 — build_corpus 경로와 중복
  chunking/        [폐기 예정 스켈레톤] 확정 설계와 불일치 (아래)
  span_index/      [폐기 예정 스켈레톤] char offset gold — 확정 설계와 불일치
scripts/
  build_corpus.py                  PDF → 구조화 JSON (메인 경로)
  pilot_semantic_vs_structure.py   1회성 파일럿(경계 절단 확인용)
config.yaml
CLAUDE.md          작업 컨텍스트(협업 AI용)
```

### 폐기 예정 스켈레톤 (미사용)

아래 모듈은 **초기 설계 흔적**이며 확정 방향과 어긋난다. 청킹 구현 착수 시
제거 후 확정 설계로 재작성한다. 현재 파이프라인은 이들을 사용하지 않는다.

- `chunking/base.py` — `SourceRef(char_start, char_end)` 좌표 기반 gold.
  확정 설계는 **청크 ID 집합 기반 containment**(좌표 불필요).
- `chunking/strategies.py` — `hierarchical()` docstring이
  "summary node + 2단계 검색". 확정 설계는 **dense 단일 단계**
  (flat 청크에 section path prepend, 2단계 검색 배제).
- `span_index/` — char offset 좌표계 gold. 위와 동일 이유로 폐기.
- `preprocessing/extract_sections.py` — 섹션별 PDF를 따로 잘라내는
  경로. 메인은 `build_corpus.py`가 페이지 범위로 직접 파싱하므로 중복.

---

## 실행

의존성:
```bash
pip install pdfplumber pyyaml
```

코퍼스 빌드:
```bash
python scripts/build_corpus.py --config config.yaml
```

- 입력: `config.yaml`의 `pdf_path` + `sections`(LCO별 시작/끝 PDF 페이지,
  1-indexed, PDF 뷰어 기준 실제 페이지 — 목차 번호 아님)
- 출력: `data/processed/`의 `sections.jsonl`, `struct_chunks.jsonl`
- 콘솔에 섹션별 조건 라벨 연속성(`A,B,C...`) 경고와 빈 페이지 목록 출력.
  새 섹션 추가 시 이 로그로 파싱이 깨졌는지 먼저 확인.
- LCO ID/제목은 페이지 헤더에서 자동 인식 — `sections`에 이름 불필요.
- 컬럼 경계(`col1_max`/`col2_max`)·헤더 여백(`header_margin`)이 페이지마다
  다르면 `layout.overrides.<lco>`에 개별 override 추가.

결과 확인:
```bash
python -c "import json; print(json.loads(open('data/processed/sections.jsonl', encoding='utf-8').readline())['content']['actions_text'])"
```

> `data/`, `outputs/`는 `.gitignore` 대상 — 산출물은 로컬 전용, 재실행 시 재생성.

---

## 제약

- **HW**: RTX 3060 12GB 1장. 임베딩은 소형(sentence-transformers 계열),
  generator는 7~8B 4-bit 양자화 또는 API, judge는 API 권장. 대규모 파인튜닝 배제.
- **분량**: 1~5p. **기간**: ~1개월.
- **환경**: conda env `kns_rag`, Python 3.11.