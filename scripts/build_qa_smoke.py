from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from kns_rag.io import load_jsonl, resolve_path, write_jsonl


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = resolve_path(ROOT, cfg["paths"].get("processed_dir", "data/processed"))
    qa_dir = resolve_path(ROOT, cfg["paths"].get("qa_dir", "data/qa"))
    out_path = resolve_path(ROOT, args.out) if args.out else qa_dir / "qa.jsonl"

    source_path = processed_dir / "hierarchical_source.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"missing source: {source_path}. Run scripts/build_corpus.py first.")

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
