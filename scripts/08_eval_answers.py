"""Stage 08: generated answers -> downstream QA metrics.

Each answer is evaluated against the human reference answer with a strict JSON
rubric for correctness, completeness, and structural relation accuracy.
"""

from __future__ import annotations

import argparse
import csv

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.downstream import (
    build_judge_messages,
    extract_json_object,
    lexical_token_f1,
    normalize_judgement,
    summarize_judgement_rows,
)
from kns_rag.io import load_jsonl, write_json, write_jsonl
from kns_rag.llm import create_chat_backend


SUMMARY_FIELDS = [
    "strategy",
    "context_token_budget",
    "qa_type",
    "n",
    "pass_rate",
    "mean_score",
    "correctness",
    "completeness",
    "relation_accuracy",
    "unsupported_claim_rate",
    "mean_lexical_f1",
    "judge_parse_error_rate",
    "error_counts",
]


def write_summary_csv(path, summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for strategy, budget_values in summary.items():
            for budget, values in budget_values.items():
                writer.writerow(
                    {
                        "strategy": strategy,
                        "context_token_budget": budget,
                        "qa_type": "ALL",
                        **{key: values[key] for key in SUMMARY_FIELDS[3:-1]},
                        "error_counts": str(values.get("error_counts") or {}),
                    }
                )
                for qa_type, type_values in values.get("by_type", {}).items():
                    writer.writerow(
                        {
                            "strategy": strategy,
                            "context_token_budget": budget,
                            "qa_type": qa_type,
                            **{key: type_values[key] for key in SUMMARY_FIELDS[3:-1]},
                            "error_counts": str(type_values.get("error_counts") or {}),
                        }
                    )


def failed_judgement(error_type: str, rationale: str) -> dict:
    return {
        "verdict": "fail",
        "passed": False,
        "correctness": 0,
        "completeness": 0,
        "relation_accuracy": 0,
        "unsupported_claim": False,
        "error_types": [error_type],
        "rationale": rationale,
        "normalized_score": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--judgements-out", default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--token-budget", type=int, action="append", dest="token_budgets")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    cfg = load_config(args.config)
    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    answers_path = cfg.resolve(args.answers) if args.answers else cfg.generated_answers_file
    judgements_path = (
        cfg.resolve(args.judgements_out) if args.judgements_out else cfg.answer_judgements_file
    )
    metrics_path = (
        cfg.resolve(args.metrics_out) if args.metrics_out else cfg.downstream_metrics_file
    )
    csv_path = cfg.resolve(args.csv_out) if args.csv_out else metrics_path.with_suffix(".csv")

    qa_by_id = {str(row.get("id")): row for row in load_jsonl(qa_path)}
    answers = load_jsonl(answers_path)
    budgets = set(args.token_budgets or [])
    selected = [
        row
        for row in answers
        if (args.strategy == "all" or row.get("strategy") == args.strategy)
        and (not budgets or int(row.get("context_token_budget") or 0) in budgets)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    settings = dict(cfg.raw.get("llm", {}).get("judge") or {})
    backend = create_chat_backend(settings, role="judge")
    mode = str(settings.get("mode") or "")

    output_rows = load_jsonl(judgements_path) if args.resume and judgements_path.exists() else []
    completed = {
        (str(row.get("experiment_id")), str(row.get("judge_model")))
        for row in output_rows
        if not row.get("judge_error") and not row.get("judge_parse_error")
    }
    pending = [
        row
        for row in selected
        if (str(row.get("experiment_id")), backend.model_name) not in completed
    ]
    print(
        f"judge={backend.model_name}, selected={len(selected)}, "
        f"completed={len(selected) - len(pending)}, pending={len(pending)}"
    )

    for index, answer_row in enumerate(pending, 1):
        qa_id = str(answer_row.get("qa_id") or "")
        qa = qa_by_id.get(qa_id)
        if qa is None:
            raise ValueError(f"answer references unknown qa_id: {qa_id}")
        reference_answer = str(qa.get("reference_answer") or "").strip()
        if not reference_answer:
            raise ValueError(f"QA record has no reference_answer: {qa_id}")
        candidate_answer = str(answer_row.get("answer") or "").strip()

        record = {
            "experiment_id": answer_row.get("experiment_id"),
            "qa_id": qa_id,
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "strategy": answer_row.get("strategy"),
            "context_token_budget": answer_row.get("context_token_budget"),
            "context_tokens": answer_row.get("context_tokens"),
            "context_count": answer_row.get("context_count"),
            "generator_model": answer_row.get("generator_model"),
            "judge_mode": mode,
            "judge_model": backend.model_name,
            "question": qa.get("question"),
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
            "lexical_f1": lexical_token_f1(reference_answer, candidate_answer),
            "judge_raw": "",
            "judge_error": None,
            "judge_parse_error": False,
        }

        if answer_row.get("generation_error") or not candidate_answer:
            judgement = failed_judgement(
                "insufficient_context",
                f"No generated answer: {answer_row.get('generation_error') or 'empty answer'}",
            )
            record["judge_error"] = "generation_failed_or_empty"
        else:
            messages = build_judge_messages(
                question=str(qa.get("question") or ""),
                reference_answer=reference_answer,
                candidate_answer=candidate_answer,
                contexts=answer_row.get("contexts") or [],
            )
            try:
                raw = backend.generate(messages)
                record["judge_raw"] = raw
                judgement = normalize_judgement(extract_json_object(raw))
            except Exception as exc:  # preserve the row and allow resume/retry
                record["judge_error"] = f"{type(exc).__name__}: {exc}"
                record["judge_parse_error"] = True
                judgement = failed_judgement("other", record["judge_error"])
                print(f"judge error: {answer_row.get('experiment_id')}: {record['judge_error']}")

        record.update(judgement)
        output_rows.append(record)
        if index % args.checkpoint_every == 0:
            write_jsonl(judgements_path, output_rows)
            print(f"checkpoint {index}/{len(pending)} -> {judgements_path}")

    write_jsonl(judgements_path, output_rows)

    selected_ids = {str(row.get("experiment_id")) for row in selected}
    metric_rows = [
        row
        for row in output_rows
        if str(row.get("experiment_id")) in selected_ids
        and str(row.get("judge_model")) == backend.model_name
    ]
    summary = summarize_judgement_rows(metric_rows)
    metrics = {
        "evaluation": {
            "judge_mode": mode,
            "judge_model": backend.model_name,
            "pass_definition": (
                "correctness=2 and completeness=2 and relation_accuracy=2 "
                "and unsupported_claim=false"
            ),
        },
        "summary": summary,
        "details": metric_rows,
    }
    write_json(metrics_path, metrics)
    write_summary_csv(csv_path, summary)
    print(f"judgements -> {judgements_path}")
    print(f"downstream metrics -> {metrics_path}")
    print(f"downstream summary csv -> {csv_path}")

    for strategy, budget_values in summary.items():
        for budget, values in budget_values.items():
            print(
                f"{strategy}@{budget}t: n={values['n']}, "
                f"pass={values['pass_rate']:.4f}, score={values['mean_score']:.4f}, "
                f"correctness={values['correctness']:.4f}, "
                f"completeness={values['completeness']:.4f}, "
                f"relation={values['relation_accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
