"""Stage 06: retrieval runs -> 지표 (outputs/eval/retrieval_metrics.json).

data/qa/qa.jsonl(사람 검증 QA)과 Stage 04의 runs.jsonl이 있어야 동작한다.
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.evaluation import evaluate_run_records
from kns_rag.io import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--runs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        help="Include unanswerable QA in retrieval metrics. Default skips them.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    runs_path = cfg.resolve(args.runs) if args.runs else cfg.retrieval_runs_file
    out_path = cfg.resolve(args.out) if args.out else cfg.retrieval_metrics_file

    qa_records = load_jsonl(qa_path)
    run_records = load_jsonl(runs_path)
    k_values = cfg.raw.get("evaluation", {}).get("k_values") or [1, 3, 5]
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
