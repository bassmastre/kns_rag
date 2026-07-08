from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from kns_rag.evaluation import evaluate_run_records
from kns_rag.io import load_jsonl, resolve_path, write_json


def default_qa_file(cfg: dict) -> Path:
    qa_dir = resolve_path(ROOT, cfg["paths"].get("qa_dir", "data/qa"))
    return qa_dir / "qa.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include unanswerable QA in retrieval metrics. Default skips them.",
    )
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = resolve_path(ROOT, cfg["paths"].get("output_dir", "outputs"))
    qa_path = resolve_path(ROOT, args.qa_file) if args.qa_file else default_qa_file(cfg)
    runs_path = resolve_path(ROOT, args.runs) if args.runs else output_dir / "retrieval" / "runs.jsonl"
    out_path = resolve_path(ROOT, args.out) if args.out else output_dir / "eval" / "retrieval_metrics.json"

    qa_records = load_jsonl(qa_path)
    run_records = load_jsonl(runs_path)
    k_values = cfg.get("evaluation", {}).get("k_values") or [1, 3, 5]
    metrics = evaluate_run_records(
        qa_records,
        run_records,
        k_values=k_values,
        skip_unanswerable=not args.include_unanswerable,
    )
    write_json(out_path, metrics)
    print(f"metrics -> {out_path}")
    for strategy, values in metrics["summary"].items():
        summary_line = ", ".join(
            [f"n={values['n']}", f"MRR={values['mrr']:.4f}"]
            + [f"hit@{k}={values[f'hit@{k}']:.4f}" for k in k_values]
        )
        print(f"{strategy}: {summary_line}")


if __name__ == "__main__":
    main()
