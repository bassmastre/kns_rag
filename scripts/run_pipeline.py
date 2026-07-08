from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = ["corpus", "chunks", "index", "retrieve", "rag_inputs", "eval"]
COMMANDS = {
    "corpus": ["scripts/build_corpus.py"],
    "chunks": ["scripts/build_chunks.py"],
    "index": ["scripts/build_index.py"],
    "retrieve": ["scripts/retrieve.py"],
    "rag_inputs": ["scripts/build_rag_inputs.py"],
    "eval": ["scripts/eval_retrieval.py"],
}


def selected_stages(start: str, stop: str) -> list[str]:
    si = STAGES.index(start)
    ei = STAGES.index(stop)
    if si > ei:
        raise ValueError("--from-stage must not come after --to-stage")
    return STAGES[si : ei + 1]


def run_stage(stage: str, *, config: str, strategy: str | None, qa_file: str | None) -> None:
    cmd = [sys.executable, *COMMANDS[stage], "--config", config]
    if stage in {"chunks", "index", "retrieve"} and strategy:
        cmd.extend(["--strategy", strategy])
    if stage in {"retrieve", "eval"} and qa_file:
        cmd.extend(["--qa-file", qa_file])

    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", default="all")
    parser.add_argument("--qa-file", default=None)
    parser.add_argument("--from-stage", choices=STAGES, default="corpus")
    parser.add_argument("--to-stage", choices=STAGES, default="eval")
    args = parser.parse_args()

    for stage in selected_stages(args.from_stage, args.to_stage):
        run_stage(stage, config=args.config, strategy=args.strategy, qa_file=args.qa_file)


if __name__ == "__main__":
    main()
