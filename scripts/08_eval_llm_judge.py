"""Evaluate generated answers with the strict binary API LLM judge.

This is the paper-facing O/X evaluator. It uses the reference answer,
evidence_keywords, and source_section, but does not expose retrieval context to
the judge. Results are checkpointed and resumable.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_json, write_jsonl
from kns_rag.llm import create_chat_backend
from kns_rag.strict_judge import (
    build_judge_messages,
    parse_judge_output,
    summarize_judgements,
)


DETAIL_FIELDS = [
    "experiment_id",
    "qa_id",
    "qa_type",
    "source_section",
    "strategy",
    "context_token_budget",
    "generator_model",
    "judge_model",
    "judge_verdict",
    "judge_reason",
    "judge_error",
    "judge_seconds",
]

SUMMARY_FIELDS = [
    "group",
    "name",
    "n",
    "O",
    "X",
    "accuracy",
    "judge_error_count",
]


def judgement_key(row: dict, model_name: str | None = None) -> tuple[str, str]:
    return (
        str(row.get("experiment_id") or ""),
        str(model_name if model_name is not None else row.get("judge_model") or ""),
    )


def answer_key(row: dict) -> tuple[str, str]:
    return (
        str(row.get("experiment_id") or ""),
        str(row.get("generator_model") or ""),
    )


def write_detail_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({"group": "overall", "name": "ALL", **summary["overall"]})
        for group in ("by_strategy", "by_strategy_budget", "by_type", "by_section"):
            for name, values in summary[group].items():
                writer.writerow({"group": group, "name": name, **values})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--answers-kind", choices=["rag", "llm_only"], default="rag")
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--summary-csv-out", default=None)
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--token-budget", type=int, action="append", dest="token_budgets")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    cfg = load_config(args.config)
    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    if args.answers:
        answers_path = cfg.resolve(args.answers)
    elif args.answers_kind == "llm_only":
        answers_path = cfg.llm_only_answers_file
    else:
        answers_path = cfg.generated_answers_file

    if args.out:
        out_path = cfg.resolve(args.out)
    elif args.answers_kind == "llm_only":
        out_path = cfg.llm_only_judgements_file
    else:
        out_path = cfg.strict_answer_judgements_file

    summary_path = (
        cfg.resolve(args.summary_out)
        if args.summary_out
        else out_path.with_name(out_path.stem + "_summary.json")
    )
    detail_csv_path = (
        cfg.resolve(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    )
    summary_csv_path = (
        cfg.resolve(args.summary_csv_out)
        if args.summary_csv_out
        else out_path.with_name(out_path.stem + "_summary.csv")
    )

    qa_by_id = {str(row.get("id")): row for row in load_jsonl(qa_path)}
    answers_by_key = {answer_key(row): row for row in load_jsonl(answers_path)}
    answers = list(answers_by_key.values())
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
    judge_mode = str(settings.get("mode") or "")

    previous_rows = load_jsonl(out_path) if args.resume and out_path.exists() else []
    output_by_key = {judgement_key(row): row for row in previous_rows}
    completed = {
        key
        for key, row in output_by_key.items()
        if row.get("judge_verdict") in {"O", "X"} and not row.get("judge_error")
    }
    pending = [
        row
        for row in selected
        if judgement_key(row, backend.model_name) not in completed
    ]

    print(
        f"judge={backend.model_name} | selected={len(selected)} | "
        f"completed={len(selected) - len(pending)} | pending={len(pending)}"
    )

    for index, answer_row in enumerate(pending, 1):
        qa_id = str(answer_row.get("qa_id") or "")
        qa = qa_by_id.get(qa_id)
        if qa is None:
            raise ValueError(f"answer references unknown qa_id: {qa_id}")

        generated_answer = str(answer_row.get("answer") or "").strip()
        record = {
            "experiment_id": answer_row.get("experiment_id"),
            "qa_id": qa_id,
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "source_section": qa.get("source_section"),
            "strategy": answer_row.get("strategy"),
            "context_token_budget": answer_row.get("context_token_budget"),
            "context_tokens": answer_row.get("context_tokens"),
            "context_count": answer_row.get("context_count"),
            "generator_model": answer_row.get("generator_model"),
            "judge_mode": judge_mode,
            "judge_model": backend.model_name,
            "question": qa.get("question"),
            "reference_answer": qa.get("reference_answer"),
            "evidence_keywords": qa.get("evidence_keywords") or [],
            "generated_answer": generated_answer,
            "judge_raw": "",
            "judge_verdict": None,
            "judge_reason": None,
            "judge_error": None,
        }

        started = time.perf_counter()
        if answer_row.get("generation_error") or not generated_answer:
            record["judge_verdict"] = "X"
            record["judge_reason"] = (
                "Rule 1: the generator failed, returned an empty answer, or did not commit to an answer."
            )
        else:
            try:
                raw = backend.generate(
                    build_judge_messages(
                        question=str(qa.get("question") or ""),
                        reference_answer=str(qa.get("reference_answer") or ""),
                        evidence_keywords=list(qa.get("evidence_keywords") or []),
                        source_section=str(qa.get("source_section") or ""),
                        generated_answer=generated_answer,
                    )
                )
                record["judge_raw"] = raw
                verdict, reason = parse_judge_output(raw)
                record["judge_verdict"] = verdict
                record["judge_reason"] = reason
            except Exception as exc:
                record["judge_error"] = f"{type(exc).__name__}: {exc}"

        record["judge_seconds"] = round(time.perf_counter() - started, 3)
        output_by_key[judgement_key(record)] = record
        print(
            f"[{index}/{len(pending)}] {record['experiment_id']} | "
            f"verdict={record['judge_verdict']} | error={record['judge_error']}"
        )

        if index % args.checkpoint_every == 0:
            rows = list(output_by_key.values())
            write_jsonl(out_path, rows)
            write_detail_csv(detail_csv_path, rows)
            graded = [row for row in rows if row.get("judge_verdict") in {"O", "X"}]
            summary = summarize_judgements(graded)
            write_json(summary_path, summary)
            write_summary_csv(summary_csv_path, summary)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    rows = list(output_by_key.values())
    write_jsonl(out_path, rows)
    write_detail_csv(detail_csv_path, rows)
    graded = [row for row in rows if row.get("judge_verdict") in {"O", "X"}]
    summary = summarize_judgements(graded)
    write_json(summary_path, summary)
    write_summary_csv(summary_csv_path, summary)

    print(f"judge results -> {out_path}")
    print(f"judge summary -> {summary_path}")
    print(f"detail csv -> {detail_csv_path}")
    print(f"summary csv -> {summary_csv_path}")


if __name__ == "__main__":
    main()
