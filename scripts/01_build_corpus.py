"""Stage 01: PDF -> 구조화 코퍼스 (raw / hierarchical_source / condition_chunks).

zero-arg 실행 가능. 01~03은 사람 개입 없이 순서대로 실행된다.
"""

from __future__ import annotations

import argparse

import kns_rag.parse as parse_mod
from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.corpus import assemble_sections, build_records
from kns_rag.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)

    pages_data, skipped = [], []
    with parse_mod.pdfplumber.open(cfg.pdf_path) as pdf:
        for sec in cfg.raw["sections"]:
            for pno in range(sec["start"], sec["end"] + 1):
                pd = parse_mod.parse_page_raw(pdf.pages[pno - 1], cfg.raw)
                if pd is None:
                    skipped.append(pno)
                    continue
                pd["page_no"] = pno
                pd["source_doc"] = cfg.raw["source_doc"]
                pages_data.append(pd)

    sections = assemble_sections(pages_data)

    hier_records, flat_records, condition_records = [], [], []
    for s in sections:
        h, f, c = build_records(s)
        hier_records.append(h)
        flat_records.extend(f)
        condition_records.extend(c)

    write_jsonl(
        cfg.processed_dir / "raw.jsonl",
        [
            {"id": r["id"], "metadata": r["metadata"], "raw_text": r["content"]["raw_text"]}
            for r in hier_records
        ],
    )
    write_jsonl(cfg.processed_dir / "hierarchical_source.jsonl", flat_records)
    write_jsonl(cfg.processed_dir / "condition_chunks.jsonl", condition_records)

    print(f"empty pages: {skipped or 'none'}")
    print(f"sections: {len(hier_records)}")
    print(f"flat chunks: {len(flat_records)}")
    print(f"condition chunks: {len(condition_records)}")
    for h in hier_records:
        letters = [cb["label"] for cb in h["content"]["condition_blocks"]]
        expected = [chr(ord("A") + i) for i in range(len(letters))]
        warn = "" if letters == expected else f" letter continuity warning: {letters}"
        print(
            f"{h['id']} pages={h['metadata']['source_pages']} "
            f"conditions={letters}{warn}"
        )


if __name__ == "__main__":
    main()
