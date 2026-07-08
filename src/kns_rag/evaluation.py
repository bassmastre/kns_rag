from __future__ import annotations

from collections import defaultdict
from typing import Any


def gold_evidence_ids(qa: dict[str, Any]) -> list[str]:
    """Accept a few QA gold field names while keeping one internal format."""
    for key in ("gold_evidence_ids", "evidence_ids", "gold_ids"):
        value = qa.get(key)
        if value:
            return [str(x) for x in value]
    gold = qa.get("gold") or {}
    value = gold.get("evidence_ids") if isinstance(gold, dict) else None
    return [str(x) for x in value] if value else []


def is_answerable(qa: dict[str, Any]) -> bool:
    value = qa.get("answerable")
    if value is not None:
        return bool(value)
    qtype = qa.get("type") or qa.get("qa_type")
    if qtype == "unanswerable":
        return False
    return bool(gold_evidence_ids(qa))


def result_hits_gold(result: dict[str, Any], gold_ids: set[str]) -> bool:
    if not gold_ids:
        return False
    ids = set(str(x) for x in (result.get("evidence_ids") or []))
    chunk_id = result.get("chunk_id")
    if chunk_id:
        ids.add(str(chunk_id))
    return bool(ids & gold_ids)


def evaluate_run_records(
    qa_records: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
    *,
    k_values: list[int],
    skip_unanswerable: bool = True,
) -> dict[str, Any]:
    qa_by_id = {q.get("id"): q for q in qa_records}
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    for run in run_records:
        qa = qa_by_id.get(run.get("qa_id"))
        if qa is None:
            continue
        if skip_unanswerable and not is_answerable(qa):
            continue

        gold_ids = set(gold_evidence_ids(qa))
        ranks = []
        for result in run.get("results", []):
            if result_hits_gold(result, gold_ids):
                ranks.append(int(result.get("rank") or 0))
        first_rank = min(ranks) if ranks else None
        row = {
            "qa_id": qa.get("id"),
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "strategy": run.get("strategy"),
            "gold_evidence_ids": sorted(gold_ids),
            "first_hit_rank": first_rank,
        }
        for k in k_values:
            row[f"hit@{k}"] = bool(first_rank is not None and first_rank <= k)
        row["rr"] = 1.0 / first_rank if first_rank else 0.0
        details.append(row)
        by_strategy[str(run.get("strategy"))].append(row)

    summary = {}
    for strategy, rows in sorted(by_strategy.items()):
        n = len(rows)
        if n == 0:
            continue
        summary[strategy] = {
            "n": n,
            "mrr": sum(r["rr"] for r in rows) / n,
            **{f"hit@{k}": sum(1 for r in rows if r[f"hit@{k}"]) / n for k in k_values},
        }

        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_type[str(r.get("qa_type") or "unknown")].append(r)
        summary[strategy]["by_type"] = {
            qtype: {
                "n": len(type_rows),
                "mrr": sum(r["rr"] for r in type_rows) / len(type_rows),
                **{
                    f"hit@{k}": sum(1 for r in type_rows if r[f"hit@{k}"]) / len(type_rows)
                    for k in k_values
                },
            }
            for qtype, type_rows in sorted(by_type.items())
        }

    return {"summary": summary, "details": details}
