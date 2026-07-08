"""QA 준비 유틸: hierarchical_source에서 스모크용 QA를 자동 생성.

선형 파이프라인 스테이지가 아니다. 생성물(data/qa/qa.jsonl)은 파이프라인
배선 확인용이며 최종 평가에 쓰지 않는다 — 최종 QA는 사람이 검증해 만든다.
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl


def action_question(record: dict) -> str | None:
    meta = record.get("metadata", {})
    content = record.get("content", {})
    lco = meta.get("lco")
    cond = content.get("condition_text")
    label = meta.get("action_label")
    if not lco or not cond or not label:
        return None
    return f"For LCO {lco}, what is Required Action {label} when the condition is: {cond}"


def completion_time_question(record: dict) -> str | None:
    meta = record.get("metadata", {})
    content = record.get("content", {})
    lco = meta.get("lco")
    label = meta.get("action_label")
    if not lco or not label or not content.get("completion_time"):
        return None
    return f"For LCO {lco}, what is the Completion Time for Required Action {label}?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = cfg.resolve(args.out) if args.out else cfg.qa_file

    source_path = cfg.processed_dir / "hierarchical_source.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing source: {source_path}. Run scripts/01_build_corpus.py first."
        )

    rows = []
    for record in load_jsonl(source_path):
        rid = record.get("id")
        meta = record.get("metadata", {})
        if meta.get("action_label") is None:
            continue

        q = action_question(record)
        if q:
            rows.append({
                "id": f"smoke_action_{len(rows)+1:03d}",
                "type": "condition_action_mapping",
                "question": q,
                "gold_evidence_ids": [rid],
                "answerable": True,
                "source": "auto_smoke",
            })
        if len(rows) >= args.limit:
            break

        q = completion_time_question(record)
        if q:
            rows.append({
                "id": f"smoke_ct_{len(rows)+1:03d}",
                "type": "extractive",
                "question": q,
                "gold_evidence_ids": [rid],
                "answerable": True,
                "source": "auto_smoke",
            })
        if len(rows) >= args.limit:
            break

    if not rows:
        raise ValueError("no action records found for smoke QA generation")

    write_jsonl(out_path, rows)
    print(f"smoke QA: {len(rows)} records -> {out_path}")
    print("WARNING: this file is for pipeline smoke testing only, not for final evaluation.")


if __name__ == "__main__":
    main()
