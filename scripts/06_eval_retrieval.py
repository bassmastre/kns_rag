"""Stage 06: retrieval runs -> 지표 (outputs/eval/retrieval_metrics.json).

data/qa/dddd.jsonl(사람 검증 QA)과 Stage 04의 runs.jsonl이 있어야 동작한다.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.evaluation import classify_hit, evaluate_run_records, evidence_keyword_groups, ranked_results
from kns_rag.io import load_jsonl, write_json


BASELINE_STRATEGIES = {"naive_fixed_length", "sliding_window", "semantic"}
DIAGNOSTIC_TOP_K = 5


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


def diagnostic_rows(qa_records, run_records):
    """Summarize genuine/coincidental keyword hits in top-5 retrieval results."""
    qa_by_id = {q.get("id"): q for q in qa_records}
    counts = defaultdict(lambda: {"n_hits": 0, "genuine": 0, "coincidental": 0})

    for run in run_records:
        qa = qa_by_id.get(run.get("qa_id"))
        if qa is None:
            continue
        strategy = str(run.get("strategy"))
        qtype = str(qa.get("type") or qa.get("qa_type") or "unknown")
        keyword_groups = evidence_keyword_groups(qa)
        evidence_units = [str(x) for x in qa["evidence_units"] if str(x).strip()]
        keys = [(strategy, "ALL"), (strategy, qtype)]
        for key in keys:
            counts[key]
        for rank, result in ranked_results(run.get("results", [])):
            if rank > DIAGNOSTIC_TOP_K:
                break
            for group in keyword_groups:
                verdict = classify_hit(result, group, evidence_units)
                if verdict == "unmatched":
                    continue
                for key in keys:
                    counts[key]["n_hits"] += 1
                    counts[key][verdict] += 1

    rows = []
    for (strategy, qtype), values in sorted(counts.items()):
        n_hits = values["n_hits"]
        genuine = values["genuine"]
        rows.append({
            "strategy": strategy,
            "qa_type": qtype,
            "n_hits": n_hits,
            "genuine": genuine,
            "coincidental": values["coincidental"],
            "genuine_rate": genuine / n_hits if n_hits else 0.0,
        })
    return rows


def write_diagnostics_csv(path, rows):
    """Write genuine/coincidental retrieval diagnostics as a CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["strategy", "qa_type", "n_hits", "genuine", "coincidental", "genuine_rate"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    diagnostics_path = cfg.output_dir / "eval" / "retrieval_metrics_diagnostics.csv"

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
    diagnostics = diagnostic_rows(qa_records, run_records)
    write_diagnostics_csv(diagnostics_path, diagnostics)
    print(f"metrics -> {out_path}")
    print(f"summary csv -> {csv_path}")
    print(f"diagnostics csv -> {diagnostics_path}")
    print(
        f"diagnostics note: {', '.join(sorted(BASELINE_STRATEGIES))} evidence_ids are regex-inferred, "
        "so their genuine/coincidental labels are approximate."
    )
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
