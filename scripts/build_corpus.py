import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import src.kns_rag.parse as parse_mod
from src.kns_rag.corpus import assemble_sections, build_records


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

config_path = Path(args.config)
if not config_path.is_absolute():
    config_path = ROOT / config_path

with config_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

pdf_path = Path(cfg["pdf_path"])
if not pdf_path.is_absolute():
    pdf_path = ROOT / pdf_path

out_dir = Path(cfg["out_dir"])
if not out_dir.is_absolute():
    out_dir = ROOT / out_dir
out_dir.mkdir(parents=True, exist_ok=True)

pages_data, skipped = [], []
with parse_mod.pdfplumber.open(pdf_path) as pdf:
    for sec in cfg["sections"]:
        for pno in range(sec["start"], sec["end"] + 1):
            page = pdf.pages[pno - 1]
            pd = parse_mod.parse_page_raw(page, cfg)
            if pd is None:
                skipped.append(pno)
                continue
            pd["page_no"] = pno
            pd["source_doc"] = cfg["source_doc"]
            pages_data.append(pd)

sections = assemble_sections(pages_data)

hier_records, flat_records = [], []
for s in sections:
    h, f = build_records(s)
    hier_records.append(h)
    flat_records.extend(f)

with (out_dir / "sections.jsonl").open("w", encoding="utf-8") as fh:
    for r in hier_records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

with (out_dir / "struct_chunks.jsonl").open("w", encoding="utf-8") as fh:
    for r in flat_records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"empty pages: {skipped or 'none'}")
print(f"sections: {len(hier_records)}")
print(f"flat chunks: {len(flat_records)}")
for h in hier_records:
    letters = [cb["label"] for cb in h["content"]["condition_blocks"]]
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    warn = "" if letters == expected else f" letter continuity warning: {letters}"
    print(
        f"{h['id']} pages={h['metadata']['source_pages']} "
        f"conditions={letters}{warn}"
    )
