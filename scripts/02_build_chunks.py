"""Stage 02: 코퍼스 -> 전략별 청크 (data/chunks/<strategy>.jsonl).

zero-arg 실행 가능. 01~03은 사람 개입 없이 순서대로 실행된다.
"""

from __future__ import annotations

import argparse

from kns_rag.chunking.strategies import DEFAULT_PARAMS, STRATEGIES, STRATEGY_INPUTS
from kns_rag.config import DEFAULT_CONFIG_PATH, ExperimentConfig, load_config
from kns_rag.io import load_jsonl, write_jsonl


def strategy_params(cfg: ExperimentConfig, strategy: str) -> dict:
    params = dict(DEFAULT_PARAMS.get(strategy, {}))
    params.update(cfg.raw.get("chunking", {}).get("params", {}).get(strategy, {}))
    if strategy == "semantic" and not params.get("model_name"):
        params["model_name"] = cfg.raw.get("embedding_model", {}).get("name")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", *sorted(STRATEGIES)],
        help="Chunking strategy to build. Use 'all' for config.chunking.strategies.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    cache: dict[str, list[dict]] = {}
    for strategy in cfg.selected_strategies(args.strategy):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown chunking strategy: {strategy}")

        input_kind = STRATEGY_INPUTS[strategy]
        if input_kind not in cache:
            input_path = cfg.processed_dir / f"{input_kind}.jsonl"
            if not input_path.exists():
                raise FileNotFoundError(
                    f"missing input for {strategy}: {input_path}. "
                    "Run scripts/01_build_corpus.py first."
                )
            cache[input_kind] = load_jsonl(input_path)

        chunks = STRATEGIES[strategy](cache[input_kind], **strategy_params(cfg, strategy))
        out_path = cfg.chunks_file(strategy)
        write_jsonl(out_path, chunks)
        print(f"{strategy}: {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
