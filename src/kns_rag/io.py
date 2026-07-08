from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL with line-numbered errors."""
    records: list[dict[str, Any]] = []
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def chunk_body(chunk: dict[str, Any]) -> str:
    """Return the only text field that may be embedded/retrieved."""
    return str(chunk.get("content", {}).get("body") or "").strip()


def evidence_ids(record: dict[str, Any]) -> list[str]:
    """Return metadata evidence ids without using them as retrievable text."""
    ids = record.get("metadata", {}).get("evidence_ids") or []
    return [str(x) for x in ids if x]
