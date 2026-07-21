"""Stage 07: RAG inputs -> generated answers.

The configured generator is loaded once and used for every strategy/budget row.
Results are checkpointed so a long local run can be resumed safely.
"""

from __future__ import annotations

import argparse

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl
from kns_rag.llm import create_chat_backend


def result_key(row: dict, model_name: str | None = None) -> tuple[str, str]:
    return (
        str(row.get("experiment_id") or ""),
        str(model_name if model_name is not None else row.get("generator_model") or ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--token-budget", type=int, action="append", dest="token_budgets")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    cfg = load_config(args.config)
    inputs_path = cfg.resolve(args.inputs) if args.inputs else cfg.rag_inputs_file
    out_path = cfg.resolve(args.out) if args.out else cfg.generated_answers_file
    settings = dict(cfg.raw.get("llm", {}).get("generator") or {})
    backend = create_chat_backend(settings, role="generator")
    mode = str(settings.get("mode") or "")

    rows = load_jsonl(inputs_path)
    budgets = set(args.token_budgets or [])
    selected = [
        row
        for row in rows
        if (args.strategy == "all" or row.get("strategy") == args.strategy)
        and (not budgets or int(row.get("context_token_budget") or 0) in budgets)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    previous_rows = load_jsonl(out_path) if args.resume and out_path.exists() else []
    output_by_key = {result_key(row): row for row in previous_rows}
    completed = {
        key
        for key, row in output_by_key.items()
        if row.get("answer") and not row.get("generation_error")
    }

    pending = [
        row
        for row in selected
        if result_key(row, backend.model_name) not in completed
    ]
    print(
        f"generator={backend.model_name}, selected={len(selected)}, "
        f"completed={len(selected) - len(pending)}, pending={len(pending)}"
    )

    for index, row in enumerate(pending, 1):
        record = {
            "experiment_id": row.get("experiment_id"),
            "qa_id": row.get("qa_id"),
            "qa_type": row.get("qa_type"),
            "strategy": row.get("strategy"),
            "question": row.get("question"),
            "context_token_budget": row.get("context_token_budget"),
            "context_tokens": row.get("context_tokens"),
            "context_count": row.get("context_count"),
            "contexts": row.get("contexts") or [],
            "generator_mode": mode,
            "generator_model": backend.model_name,
        }
        try:
            answer = backend.generate(row.get("messages") or [])
            record["answer"] = answer
            record["generation_error"] = None
        except Exception as exc:  # keep long experiment runs resumable
            record["answer"] = ""
            record["generation_error"] = f"{type(exc).__name__}: {exc}"
            print(f"generation error: {row.get('experiment_id')}: {record['generation_error']}")
        output_by_key[result_key(record)] = record

        if index % args.checkpoint_every == 0:
            write_jsonl(out_path, list(output_by_key.values()))
            print(f"checkpoint {index}/{len(pending)} -> {out_path}")

    output_rows = list(output_by_key.values())
    write_jsonl(out_path, output_rows)
    success_count = sum(bool(row.get("answer")) and not row.get("generation_error") for row in output_rows)
    print(f"answers -> {out_path} ({success_count}/{len(output_rows)} successful rows)")


if __name__ == "__main__":
    main()
