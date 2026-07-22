"""Generate the forced-answer, no-context baseline for every QA record.

The configured generator is reused from config.yaml. No retrieval result or
context is supplied. The model must commit to its best answer. Results are
checkpointed and resumable.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from tqdm import tqdm

from kns_rag.config import DEFAULT_CONFIG_PATH, load_config
from kns_rag.io import load_jsonl, write_jsonl
from kns_rag.llm import create_chat_backend


FORCED_ANSWER_SYSTEM_PROMPT = """Answer each question using only your internal model knowledge. No retrieved context is available.

You must commit to your best answer. Do not refuse, abstain, ask for more context, output INSUFFICIENT_CONTEXT, or use placeholders such as [Condition X] or [Action Y]. Do not tell the user to consult another document.

Give a direct and complete answer. State the applicable section number. Include every mandatory AND-bound requirement, every OR alternative requested by the question, the correct Condition letter and logical structure, and all required numeric values, inequalities, units, actions, and Completion Times. When uncertain, select the single most likely answer and state it without qualification.
"""


def build_forced_answer_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FORCED_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question:\n{question.strip()}"},
    ]


def result_key(row: dict, model_name: str | None = None) -> tuple[str, str]:
    return (
        str(row.get("experiment_id") or ""),
        str(model_name if model_name is not None else row.get("generator_model") or ""),
    )


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_llm_only_forced")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
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
    out_path = cfg.resolve(args.out) if args.out else cfg.llm_only_answers_file
    log_path = cfg.resolve(args.log_file) if args.log_file else out_path.with_suffix(".log")
    logger = configure_logger(log_path)

    settings = dict(cfg.raw.get("llm", {}).get("generator") or {})
    backend = create_chat_backend(settings, role="generator")
    mode = str(settings.get("mode") or "")

    qa_rows = load_jsonl(qa_path)
    if args.limit is not None:
        qa_rows = qa_rows[: args.limit]

    previous_rows = load_jsonl(out_path) if args.resume and out_path.exists() else []
    output_by_key = {result_key(row): row for row in previous_rows}
    completed = {
        key
        for key, row in output_by_key.items()
        if row.get("answer") and not row.get("generation_error")
    }
    pending = [
        qa
        for qa in qa_rows
        if result_key(
            {"experiment_id": f"{qa.get('id')}::llm_only_forced"},
            backend.model_name,
        )
        not in completed
    ]

    print(
        f"model={backend.model_name} | selected={len(qa_rows)} | "
        f"completed={len(qa_rows) - len(pending)} | pending={len(pending)} | "
        f"log={log_path}"
    )
    logger.info(
        "forced-answer generation started | model=%s | selected=%d | pending=%d",
        backend.model_name,
        len(qa_rows),
        len(pending),
    )

    generation_times: list[float] = []
    success_count = 0
    failure_count = 0
    progress = tqdm(
        pending,
        total=len(pending),
        desc="Generating forced LLM-only",
        unit="answer",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    for index, qa in enumerate(progress, 1):
        qa_id = str(qa.get("id") or "")
        experiment_id = f"{qa_id}::llm_only_forced"
        record = {
            "experiment_id": experiment_id,
            "qa_id": qa_id,
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "strategy": "llm_only",
            "baseline_mode": "forced_answer",
            "question": qa.get("question"),
            "source_section": qa.get("source_section"),
            "context_token_budget": 0,
            "context_tokens": 0,
            "context_count": 0,
            "contexts": [],
            "generator_mode": mode,
            "generator_model": backend.model_name,
        }
        started = time.perf_counter()
        try:
            record["answer"] = backend.generate(
                build_forced_answer_messages(str(qa.get("question") or ""))
            )
            record["generation_error"] = None
            success_count += 1
        except Exception as exc:
            cause = exc.__cause__
            detail = f"{type(exc).__name__}: {exc}"
            if cause:
                detail += f" | cause: {type(cause).__name__}: {cause}"
            record["answer"] = ""
            record["generation_error"] = detail
            failure_count += 1
            logger.exception(
                "generation failed | experiment_id=%s | error=%s",
                experiment_id,
                detail,
            )

        generation_seconds = time.perf_counter() - started
        record["generation_seconds"] = round(generation_seconds, 3)
        generation_times.append(generation_seconds)
        output_by_key[result_key(record)] = record

        average_seconds = sum(generation_times) / len(generation_times)
        estimated_remaining = average_seconds * (len(pending) - index)
        progress.set_postfix(
            {
                "last": f"{generation_seconds:.1f}s",
                "avg": f"{average_seconds:.1f}s",
                "left": format_duration(estimated_remaining),
                "ok": success_count,
                "err": failure_count,
            },
            refresh=True,
        )
        logger.info(
            "generation completed | %d/%d | experiment_id=%s | seconds=%.3f | success=%s",
            index,
            len(pending),
            experiment_id,
            generation_seconds,
            record["generation_error"] is None,
        )

        if index % args.checkpoint_every == 0:
            write_jsonl(out_path, list(output_by_key.values()))
            logger.info("checkpoint saved | %d/%d | output=%s", index, len(pending), out_path)

    progress.close()
    write_jsonl(out_path, list(output_by_key.values()))
    print()
    print(f"완료: {success_count}건 성공, {failure_count}건 실패")
    print(f"forced LLM-only answers -> {out_path}")
    print(f"log -> {log_path}")


if __name__ == "__main__":
    main()
