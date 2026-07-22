"""Stage 08: grade generated answers with an API LLM judge.

The exact strict O/X prompt is defined in kns_rag.prompts.JUDGE_PROMPT_TEMPLATE.
The script checkpoints every record and writes JSON and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_json, write_jsonl
from kns_rag.judging import parse_judge_output, summarize_judgments
from kns_rag.llm_client import ChatClient, chat_config_from_mapping
from kns_rag.prompts import make_judge_prompt


def write_detail_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "qa_id",
        "qa_type",
        "source_section",
        "strategy",
        "judge_verdict",
        "judge_reason",
        "judge_error",
        "judge_model",
        "judge_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--provider", default=None, choices=["openai_compatible", "anthropic"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    answers_path = cfg.resolve(args.answers) if args.answers else cfg.rag_answers_file
    out_path = cfg.resolve(args.out) if args.out else cfg.judge_results_file
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    csv_path = out_path.with_suffix(".csv")

    judge_settings = cfg.raw.get("llm", {}).get("judge", {})
    client_config = chat_config_from_mapping(
        judge_settings,
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    client = ChatClient(client_config)

    qa_by_id = {str(row.get("id")): row for row in load_jsonl(qa_path)}
    answer_rows = load_jsonl(answers_path)
    existing_rows = load_jsonl(out_path) if out_path.exists() and not args.no_resume else []
    results_by_id = {
        str(row.get("experiment_id")): row
        for row in existing_rows
        if str(row.get("experiment_id") or "").strip()
    }

    attempted = 0
    total = len(answer_rows)
    for index, answer_row in enumerate(answer_rows, 1):
        row_id = str(answer_row.get("experiment_id") or "").strip()
        if not row_id:
            raise ValueError(f"answer row {index} is missing experiment_id")
        existing = results_by_id.get(row_id)
        if existing and existing.get("judge_verdict") in {"O", "X"} and not existing.get("judge_error"):
            print(f"[{index}/{total}] skip {row_id}")
            continue
        if args.limit is not None and attempted >= args.limit:
            break
        attempted += 1

        qa_id = str(answer_row.get("qa_id") or "")
        qa = qa_by_id.get(qa_id)
        if qa is None:
            raise KeyError(f"QA not found for answer {row_id}: {qa_id}")

        generated_answer = str(answer_row.get("answer") or "").strip()
        result = dict(answer_row)
        result.update(
            {
                "qa_type": qa.get("type") or qa.get("qa_type"),
                "source_section": qa.get("source_section"),
                "judge_provider": client_config.provider,
                "judge_model": client_config.model,
            }
        )

        if answer_row.get("generation_error") or not generated_answer:
            result.update(
                {
                    "judge_verdict": "X",
                    "judge_reason": "Rule 1: the generator failed or did not commit to an answer.",
                    "judge_raw_response": None,
                    "judge_error": None,
                    "judge_seconds": 0.0,
                    "judge_usage": {},
                }
            )
            print(f"[{index}/{total}] deterministic X {row_id}")
        else:
            prompt = make_judge_prompt(
                question=str(qa.get("question") or ""),
                reference_answer=str(qa.get("reference_answer") or ""),
                evidence_keywords=list(qa.get("evidence_keywords") or []),
                source_section=str(qa.get("source_section") or ""),
                generated_answer=generated_answer,
            )
            started = time.perf_counter()
            try:
                response = client.complete(prompt)
                verdict, reason = parse_judge_output(response.text)
                result.update(
                    {
                        "judge_verdict": verdict,
                        "judge_reason": reason,
                        "judge_raw_response": response.text,
                        "judge_error": None,
                        "judge_seconds": round(time.perf_counter() - started, 3),
                        "judge_usage": response.usage,
                    }
                )
                print(f"[{index}/{total}] {verdict} {row_id}")
            except Exception as exc:
                result.update(
                    {
                        "judge_verdict": None,
                        "judge_reason": None,
                        "judge_raw_response": None,
                        "judge_error": f"{type(exc).__name__}: {exc}",
                        "judge_seconds": round(time.perf_counter() - started, 3),
                        "judge_usage": {},
                    }
                )
                print(f"[{index}/{total}] error {row_id}: {exc}")

        results_by_id[row_id] = result
        rows = list(results_by_id.values())
        write_jsonl(out_path, rows)
        write_json(summary_path, summarize_judgments(rows))
        write_detail_csv(csv_path, rows)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    rows = list(results_by_id.values())
    write_jsonl(out_path, rows)
    write_json(summary_path, summarize_judgments(rows))
    write_detail_csv(csv_path, rows)
    print(f"judge results -> {out_path}")
    print(f"judge summary -> {summary_path}")
    print(f"judge csv -> {csv_path}")


if __name__ == "__main__":
    main()
