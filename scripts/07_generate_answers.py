"""Stage 07: RAG inputs -> generated answers.

The configured generator is loaded once and used for every strategy/budget row.
Results are checkpointed so a long local run can be resumed safely.
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


def result_key(row: dict, model_name: str | None = None) -> tuple[str, str]:
    return (
        str(row.get("experiment_id") or ""),
        str(
            model_name
            if model_name is not None
            else row.get("generator_model") or ""
        ),
    )


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("generate_answers")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )

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
    parser.add_argument("--inputs", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--strategy", default="all")
    parser.add_argument(
        "--token-budget",
        type=int,
        action="append",
        dest="token_budgets",
    )
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

    inputs_path = (
        cfg.resolve(args.inputs)
        if args.inputs
        else cfg.rag_inputs_file
    )
    out_path = (
        cfg.resolve(args.out)
        if args.out
        else cfg.generated_answers_file
    )

    log_path = (
        cfg.resolve(args.log_file)
        if args.log_file
        else out_path.with_suffix(".log")
    )
    logger = configure_logger(log_path)

    settings = dict(
        cfg.raw.get("llm", {}).get("generator") or {}
    )
    backend = create_chat_backend(settings, role="generator")
    mode = str(settings.get("mode") or "")

    rows = load_jsonl(inputs_path)
    budgets = set(args.token_budgets or [])

    selected = [
        row
        for row in rows
        if (
            args.strategy == "all"
            or row.get("strategy") == args.strategy
        )
        and (
            not budgets
            or int(row.get("context_token_budget") or 0) in budgets
        )
    ]

    if args.limit is not None:
        selected = selected[: args.limit]

    previous_rows = (
        load_jsonl(out_path)
        if args.resume and out_path.exists()
        else []
    )

    output_by_key = {
        result_key(row): row
        for row in previous_rows
    }

    completed = {
        key
        for key, row in output_by_key.items()
        if row.get("answer")
        and not row.get("generation_error")
    }

    pending = [
        row
        for row in selected
        if result_key(row, backend.model_name) not in completed
    ]

    logger.info(
        "generation started | model=%s | selected=%d | "
        "completed=%d | pending=%d | inputs=%s | output=%s",
        backend.model_name,
        len(selected),
        len(selected) - len(pending),
        len(pending),
        inputs_path,
        out_path,
    )

    print(
        f"model={backend.model_name} | "
        f"pending={len(pending)} | "
        f"log={log_path}"
    )

    run_started_at = time.perf_counter()
    generation_times: list[float] = []
    current_success_count = 0
    current_failure_count = 0

    progress = tqdm(
        pending,
        total=len(pending),
        desc="Generating",
        unit="answer",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    for index, row in enumerate(progress, 1):
        record = {
            "experiment_id": row.get("experiment_id"),
            "qa_id": row.get("qa_id"),
            "qa_type": row.get("qa_type"),
            "strategy": row.get("strategy"),
            "question": row.get("question"),
            "context_token_budget": row.get(
                "context_token_budget"
            ),
            "context_tokens": row.get("context_tokens"),
            "context_count": row.get("context_count"),
            "contexts": row.get("contexts") or [],
            "generator_mode": mode,
            "generator_model": backend.model_name,
        }

        generation_started_at = time.perf_counter()

        try:
            answer = backend.generate(
                row.get("messages") or []
            )
            record["answer"] = answer
            record["generation_error"] = None
            current_success_count += 1

        except Exception as exc:
            record["answer"] = ""

            cause = exc.__cause__
            detail = f"{type(exc).__name__}: {exc}"

            if cause:
                detail += (
                    f" | cause: "
                    f"{type(cause).__name__}: {cause}"
                )

            record["generation_error"] = detail
            current_failure_count += 1

            logger.exception(
                "generation failed | experiment_id=%s | error=%s",
                row.get("experiment_id"),
                detail,
            )

        generation_seconds = (
            time.perf_counter()
            - generation_started_at
        )

        record["generation_seconds"] = round(
            generation_seconds,
            3,
        )
        generation_times.append(generation_seconds)

        output_by_key[result_key(record)] = record

        average_seconds = (
            sum(generation_times) / len(generation_times)
        )
        remaining_count = len(pending) - index
        estimated_remaining = (
            average_seconds * remaining_count
        )

        progress.set_postfix(
            {
                "last": f"{generation_seconds:.1f}s",
                "avg": f"{average_seconds:.1f}s",
                "left": format_duration(
                    estimated_remaining
                ),
                "ok": current_success_count,
                "err": current_failure_count,
            },
            refresh=True,
        )

        logger.info(
            "generation completed | %d/%d | "
            "experiment_id=%s | seconds=%.3f | success=%s",
            index,
            len(pending),
            row.get("experiment_id"),
            generation_seconds,
            record["generation_error"] is None,
        )

        if index % args.checkpoint_every == 0:
            write_jsonl(
                out_path,
                list(output_by_key.values()),
            )
            logger.info(
                "checkpoint saved | %d/%d | output=%s",
                index,
                len(pending),
                out_path,
            )

    progress.close()

    output_rows = list(output_by_key.values())
    write_jsonl(out_path, output_rows)

    total_seconds = time.perf_counter() - run_started_at

    average_seconds = (
        sum(generation_times) / len(generation_times)
        if generation_times
        else 0.0
    )

    sorted_times = sorted(generation_times)

    if not sorted_times:
        median_seconds = 0.0
    elif len(sorted_times) % 2 == 1:
        median_seconds = sorted_times[
            len(sorted_times) // 2
        ]
    else:
        middle = len(sorted_times) // 2
        median_seconds = (
            sorted_times[middle - 1]
            + sorted_times[middle]
        ) / 2

    success_count = sum(
        bool(row.get("answer"))
        and not row.get("generation_error")
        for row in output_rows
    )

    logger.info(
        "generation finished | total_seconds=%.3f | "
        "average_seconds=%.3f | median_seconds=%.3f | "
        "success=%d | total_rows=%d",
        total_seconds,
        average_seconds,
        median_seconds,
        success_count,
        len(output_rows),
    )

    print()
    print(
        f"완료: {current_success_count}건 성공, "
        f"{current_failure_count}건 실패"
    )
    print(
        f"총 시간: {format_duration(total_seconds)} | "
        f"평균: {average_seconds:.2f}초/건 | "
        f"중앙값: {median_seconds:.2f}초/건"
    )
    print(f"결과: {out_path}")
    print(f"로그: {log_path}")


if __name__ == "__main__":
    main()