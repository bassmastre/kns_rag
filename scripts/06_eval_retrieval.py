"""Stage 06: retrieval runs -> 지표 (outputs/eval/retrieval_metrics.json).

data/qa/qa.jsonl(사람 검증 QA)과 Stage 04의 runs.jsonl이 있어야 동작한다.
"""

from __future__ import annotations

import argparse
import csv

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.evaluation import evaluate_run_records
from kns_rag.io import load_jsonl, write_json


def write_summary_csv(path, metrics, k_values):
    """Write overall and per-type retrieval metrics as a CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["strategy", "qa_type", "n", "mrr"]
        + [f"hit@{k}" for k in k_values]
        + ["set_recall_mrr"]
        + [f"set_recall@{k}" for k in k_values]
        + [f"recall@{k}" for k in k_values]
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, values in metrics["summary"].items():
            writer.writerow({
                "strategy": strategy,
                "qa_type": "ALL",
                "n": values["n"],
                "mrr": values["mrr"],
                **{f"hit@{k}": values[f"hit@{k}"] for k in k_values},
                "set_recall_mrr": values["set_recall_mrr"],
                **{f"set_recall@{k}": values[f"set_recall@{k}"] for k in k_values},
                **{f"recall@{k}": values[f"recall@{k}"] for k in k_values},
            })
            for qtype, type_values in values.get("by_type", {}).items():
                writer.writerow({
                    "strategy": strategy,
                    "qa_type": qtype,
                    "n": type_values["n"],
                    "mrr": type_values["mrr"],
                    **{f"hit@{k}": type_values[f"hit@{k}"] for k in k_values},
                    "set_recall_mrr": type_values["set_recall_mrr"],
                    **{f"set_recall@{k}": type_values[f"set_recall@{k}"] for k in k_values},
                    **{f"recall@{k}": type_values[f"recall@{k}"] for k in k_values},
                })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    runs_path = cfg.resolve(args.runs) if args.runs else cfg.retrieval_runs_file
    out_path = cfg.resolve(args.out) if args.out else cfg.retrieval_metrics_file
    csv_path = cfg.resolve(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")

    qa_records = load_jsonl(qa_path)
    run_records = load_jsonl(runs_path)
    k_values = cfg.raw.get("evaluation", {}).get("k_values") or [1, 3, 5]
    metrics = evaluate_run_records(
        qa_records,
        run_records,
        k_values=k_values,
    )
    write_json(out_path, metrics)
    write_summary_csv(csv_path, metrics, k_values)
    print(f"metrics -> {out_path}")
    print(f"summary csv -> {csv_path}")
    for strategy, values in metrics["summary"].items():
        summary_line = ", ".join(
            [f"n={values['n']}", f"MRR={values['mrr']:.4f}"]
            + [f"hit@{k}={values[f'hit@{k}']:.4f}" for k in k_values]
            + [f"set_recall_mrr={values['set_recall_mrr']:.4f}"]
            + [f"set_recall@{k}={values[f'set_recall@{k}']:.4f}" for k in k_values]
            + [f"recall@{k}={values[f'recall@{k}']:.4f}" for k in k_values]
        )
        print(f"{strategy}: {summary_line}")


if __name__ == "__main__":
    main()
