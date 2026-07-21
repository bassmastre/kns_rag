from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 01~03은 사람 개입 없이 순서 실행 가능. retrieve(04)부터는
# data/qa/dddd.jsonl(사람 검증 QA)이 있어야 동작한다.
STAGES = [
    "corpus",
    "chunks",
    "index",
    "retrieve",
    "rag_inputs",
    "retrieval_eval",
    "generate",
    "downstream_eval",
]
COMMANDS = {
    "corpus": ["scripts/01_build_corpus.py"],
    "chunks": ["scripts/02_build_chunks.py"],
    "index": ["scripts/03_build_index.py"],
    "retrieve": ["scripts/04_retrieve.py"],
    "rag_inputs": ["scripts/05_build_rag_inputs.py"],
    "retrieval_eval": ["scripts/06_eval_retrieval.py"],
    "generate": ["scripts/07_generate_answers.py"],
    "downstream_eval": ["scripts/08_eval_answers.py"],
}


def selected_stages(start: str, stop: str) -> list[str]:
    si = STAGES.index(start)
    ei = STAGES.index(stop)
    if si > ei:
        raise ValueError("--from-stage must not come after --to-stage")
    return STAGES[si : ei + 1]


def run_stage(stage: str, *, config: str, strategy: str | None, qa_file: str | None) -> None:
    cmd = [sys.executable, *COMMANDS[stage], "--config", config]
    if stage in {"chunks", "index", "retrieve", "generate", "downstream_eval"} and strategy:
        cmd.extend(["--strategy", strategy])
    if stage in {"retrieve", "retrieval_eval", "downstream_eval"} and qa_file:
        cmd.extend(["--qa-file", qa_file])

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--from-stage", choices=STAGES, default="corpus")
    parser.add_argument("--to-stage", choices=STAGES, default="downstream_eval")
    args = parser.parse_args()

    for stage in selected_stages(args.from_stage, args.to_stage):
        run_stage(stage, config=args.config, strategy=args.strategy, qa_file=args.qa_file)


if __name__ == "__main__":
    main()
