"""Stage 07: generate RAG answers or the no-context LLM-only baseline.

Examples:
  python scripts/07_generate_answers.py --mode rag
  python scripts/07_generate_answers.py --mode llm_only

The script checkpoints after every answer and resumes completed records by default.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl
from kns_rag.llm_client import ChatClient, chat_config_from_mapping
from kns_rag.prompts import make_llm_only_prompt, make_rag_prompt


def experiment_id(row: dict[str, Any], *, mode: str) -> str:
    existing = str(row.get("experiment_id") or "").strip()
    if existing:
        return existing
    qa_id = str(row.get("qa_id") or row.get("id") or "unknown")
    if mode == "llm_only":
        return f"{qa_id}::llm_only"
    strategy = str(row.get("strategy") or "unknown")
    if row.get("context_token_budget") is not None:
        suffix = f"{row['context_token_budget']}t"
    else:
        suffix = f"k{row.get('context_k', row.get('context_count', 'unknown'))}"
    return f"{qa_id}::{strategy}::{suffix}"


def load_inputs(*, mode: str, qa_path: Path, rag_inputs_path: Path) -> list[dict[str, Any]]:
    if mode == "rag":
        rows = load_jsonl(rag_inputs_path)
        for row in rows:
            if not str(row.get("prompt") or "").strip():
                row["prompt"] = make_rag_prompt(
                    str(row.get("question") or ""), list(row.get("contexts") or [])
                )
        return rows

    rows = []
    for qa in load_jsonl(qa_path):
        question = str(qa.get("question") or "").strip()
        rows.append(
            {
                "experiment_id": f"{qa.get('id')}::llm_only",
                "qa_id": qa.get("id"),
                "qa_type": qa.get("type") or qa.get("qa_type"),
                "strategy": "llm_only",
                "question": question,
                "source_section": qa.get("source_section"),
                "context_token_budget": 0,
                "context_tokens": 0,
                "context_count": 0,
                "contexts": [],
                "prompt": make_llm_only_prompt(question),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=["rag", "llm_only"], default="rag")
    parser.add_argument("--input", default=None, help="RAG input JSONL; only used in rag mode.")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--provider", default=None, choices=["openai_compatible", "anthropic"])
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    qa_path = cfg.resolve(args.qa_file) if args.qa_file else cfg.qa_file
    rag_inputs_path = cfg.resolve(args.input) if args.input else cfg.rag_inputs_file
    if args.out:
        out_path = cfg.resolve(args.out)
    else:
        out_path = cfg.rag_answers_file if args.mode == "rag" else cfg.llm_only_answers_file

    generator_settings = cfg.raw.get("llm", {}).get("generator", {})
    client_config = chat_config_from_mapping(
        generator_settings,
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    client = ChatClient(client_config)

    inputs = load_inputs(mode=args.mode, qa_path=qa_path, rag_inputs_path=rag_inputs_path)
    existing_rows = load_jsonl(out_path) if out_path.exists() and not args.no_resume else []
    results_by_id = {
        str(row.get("experiment_id")): row
        for row in existing_rows
        if str(row.get("experiment_id") or "").strip()
    }

    attempted = 0
    total = len(inputs)
    for index, input_row in enumerate(inputs, 1):
        row_id = experiment_id(input_row, mode=args.mode)
        existing = results_by_id.get(row_id)
        if existing and existing.get("answer") and not existing.get("generation_error"):
            print(f"[{index}/{total}] skip {row_id}")
            continue
        if args.limit is not None and attempted >= args.limit:
            break
        attempted += 1

        base_record = {key: value for key, value in input_row.items() if key != "prompt"}
        base_record["experiment_id"] = row_id
        base_record.setdefault("qa_id", input_row.get("id"))
        base_record.setdefault("strategy", "llm_only" if args.mode == "llm_only" else None)
        base_record["generator_mode"] = client_config.provider
        base_record["generator_model"] = client_config.model

        started = time.perf_counter()
        try:
            response = client.complete(str(input_row.get("prompt") or ""))
            base_record.update(
                {
                    "answer": response.text,
                    "generation_error": None,
                    "generation_seconds": round(time.perf_counter() - started, 3),
                    "generation_usage": response.usage,
                }
            )
            print(f"[{index}/{total}] ok {row_id}")
        except Exception as exc:  # keep the batch resumable while recording the exact failure
            base_record.update(
                {
                    "answer": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "generation_seconds": round(time.perf_counter() - started, 3),
                    "generation_usage": {},
                }
            )
            print(f"[{index}/{total}] error {row_id}: {exc}")

        results_by_id[row_id] = base_record
        write_jsonl(out_path, list(results_by_id.values()))

    write_jsonl(out_path, list(results_by_id.values()))
    print(f"answers -> {out_path}")


if __name__ == "__main__":
    main()
