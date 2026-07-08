import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from kns_rag.chunking.strategies import DEFAULT_PARAMS, STRATEGIES, STRATEGY_INPUTS


INPUT_FILES = {
    "raw": "raw.jsonl",
    "hierarchical_source": "hierarchical_source.jsonl",
    "condition_chunks": "condition_chunks.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def selected_strategies(cfg: dict, strategy_arg: str) -> list[str]:
    configured = cfg.get("chunking", {}).get("strategies") or list(STRATEGIES)
    if strategy_arg == "all":
        return configured
    return [strategy_arg]


def strategy_params(cfg: dict, strategy: str) -> dict:
    params = dict(DEFAULT_PARAMS.get(strategy, {}))
    params.update(cfg.get("chunking", {}).get("params", {}).get(strategy, {}))
    if strategy == "semantic" and not params.get("model_name"):
        params["model_name"] = cfg.get("embedding_model", {}).get("name")
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", *sorted(STRATEGIES)],
        help="Chunking strategy to build. Use 'all' for config.chunking.strategies.",
    )
    args = parser.parse_args()

    config_path = resolve_path(ROOT, args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = resolve_path(ROOT, cfg["paths"]["processed_dir"])
    chunks_dir = resolve_path(ROOT, cfg["paths"].get("chunks_dir", "data/chunks"))
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, list[dict]] = {}
    for strategy in selected_strategies(cfg, args.strategy):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown chunking strategy: {strategy}")

        input_kind = STRATEGY_INPUTS[strategy]
        if input_kind not in cache:
            input_path = processed_dir / INPUT_FILES[input_kind]
            if not input_path.exists():
                raise FileNotFoundError(
                    f"missing input for {strategy}: {input_path}. "
                    "Run scripts/build_corpus.py first."
                )
            cache[input_kind] = load_jsonl(input_path)

        chunks = STRATEGIES[strategy](cache[input_kind], **strategy_params(cfg, strategy))
        out_path = chunks_dir / f"{strategy}.jsonl"
        write_jsonl(out_path, chunks)
        print(f"{strategy}: {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
