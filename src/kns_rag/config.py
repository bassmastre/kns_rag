"""실험 config 로딩과 경로 계산의 단일 지점.

모든 스크립트는 여기의 load_config()로 config.yaml을 읽고, 경로는
ExperimentConfig의 path helper로만 계산한다. 상대 경로는 config 파일이
있는 디렉터리(=repo 루트) 기준으로 해석하므로 cwd에 의존하지 않는다.

outputs/ 하위 서브디렉터리 이름(indexes, retrieval, generation, eval)은
config 키로 빼지 않고 여기 한 곳에만 상수로 둔다(과설계 방지 — 바꿀 일이
생기면 이 파일만 고치면 된다).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Parsed config.yaml + path helpers. 실험 로직은 담지 않는다."""

    root: Path
    raw: dict[str, Any]

    def resolve(self, value: str | Path) -> Path:
        """Resolve a possibly-relative path against the config directory."""
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def _path(self, key: str, default: str) -> Path:
        return self.resolve(self.raw.get("paths", {}).get(key, default))

    # --- 입력/파생 (data/) ---
    @property
    def pdf_path(self) -> Path:
        return self.resolve(self.raw["pdf_path"])

    @property
    def processed_dir(self) -> Path:
        return self._path("processed_dir", "data/processed")

    @property
    def chunks_dir(self) -> Path:
        return self._path("chunks_dir", "data/chunks")

    @property
    def qa_dir(self) -> Path:
        return self._path("qa_dir", "data/qa")

    @property
    def qa_file(self) -> Path:
        """사람 검증 QA(JSONL). 04 이후 스테이지의 입력 경계."""
        return self.qa_dir / "qa.jsonl"

    # --- 실험 출력 (outputs/) ---
    @property
    def output_dir(self) -> Path:
        return self._path("output_dir", "outputs")

    @property
    def indexes_dir(self) -> Path:
        return self.output_dir / "indexes"

    def index_dir(self, strategy: str) -> Path:
        return self.indexes_dir / strategy

    def chunks_file(self, strategy: str) -> Path:
        return self.chunks_dir / f"{strategy}.jsonl"

    @property
    def retrieval_runs_file(self) -> Path:
        return self.output_dir / "retrieval" / "runs.jsonl"

    @property
    def rag_inputs_file(self) -> Path:
        return self.output_dir / "generation" / "rag_inputs.jsonl"

    @property
    def retrieval_metrics_file(self) -> Path:
        return self.output_dir / "eval" / "retrieval_metrics.json"

    # --- 전략 선택 ---
    def selected_strategies(self, strategy_arg: str = "all") -> list[str]:
        """'all'이면 config.chunking.strategies, 아니면 [strategy_arg]."""
        if strategy_arg == "all":
            return list(self.raw.get("chunking", {}).get("strategies") or [])
        return [strategy_arg]


# editable install(src-layout) 기준 repo 루트의 config.yaml.
# 스크립트 argparse의 --config 기본값으로 쓰여 어느 cwd에서든 zero-arg 실행 가능.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig(root=config_path.parent, raw=raw)
