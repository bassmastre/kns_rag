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


def gold_keyword_groups(qa: dict[str, Any]) -> list[list[str]]:
    """Return gold keyword groups where every keyword in a group must match."""
    value = qa.get("gold_keywords")
    if value is None:
        value = qa.get("gold_keyword")
    if value is None:
        gold = qa.get("gold") or {}
        value = gold.get("keywords") if isinstance(gold, dict) else None
    if not value:
        return []
    if isinstance(value, str):
        return [[value]]
    groups = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                group = [item]
            elif isinstance(item, list):
                group = [str(x) for x in item if str(x).strip()]
            else:
                group = [str(item)] if str(item).strip() else []
            if group:
                groups.append(group)
    return groups


def is_answerable(qa: dict[str, Any]) -> bool:
    value = qa.get("answerable")
    if value is not None:
        return bool(value)
    qtype = qa.get("type") or qa.get("qa_type")
    if qtype == "unanswerable":
        return False
    return bool(gold_keyword_groups(qa) or gold_evidence_ids(qa))


def normalize_match_text(text: str) -> str:
    """Normalize text for case-insensitive keyword containment checks."""
    return " ".join(str(text).lower().split())


def result_hits_gold_keywords(result: dict[str, Any], keyword_groups: list[list[str]]) -> bool:
    """Return whether a retrieved result contains any full gold keyword group."""
    return any(result_hits_keyword_group(result, group) for group in keyword_groups)


def result_hits_keyword_group(result: dict[str, Any], keyword_group: list[str]) -> bool:
    """Return whether a retrieved result contains every keyword in one group."""
    if not keyword_group:
        return False
    body = normalize_match_text(result.get("body") or "")
    if not body:
        return False
    return all(normalize_match_text(keyword) in body for keyword in keyword_group)


def result_hits_gold(result: dict[str, Any], gold_ids: set[str]) -> bool:
    if not gold_ids:
        return False
    ids = result_evidence_id_set(result)
    return bool(ids & gold_ids)


def result_evidence_id_set(result: dict[str, Any]) -> set[str]:
    """
    Return all evidence identifiers represented by a retrieval result.
    Include both result['evidence_ids'] and result['chunk_id'].
    """
    ids = {str(x) for x in (result.get("evidence_ids") or []) if str(x).strip()}
    chunk_id = result.get("chunk_id")
    if chunk_id and str(chunk_id).strip():
        ids.add(str(chunk_id))
    return ids


def all_gold_stats(
    results: list[dict[str, Any]],
    gold_ids: set[str],
    keyword_groups: list[list[str]],
    k_values: list[int],
) -> dict[str, Any]:
    """Compute cumulative all-gold and gold-coverage metrics."""
    gold_count = len(keyword_groups) if keyword_groups else len(gold_ids)
    stats: dict[str, Any] = {"gold_count": gold_count}
    all_gold_rank = None
    cumulative: set[str] = set()
    matched_groups: set[int] = set()
    ranked = sorted(
        enumerate(results, 1),
        key=lambda pair: int(pair[1].get("rank") or pair[0]),
    )

    for fallback_rank, result in ranked:
        rank = int(result.get("rank") or fallback_rank)
        if keyword_groups:
            for i, group in enumerate(keyword_groups):
                if result_hits_keyword_group(result, group):
                    matched_groups.add(i)
            if all_gold_rank is None and len(matched_groups) == gold_count:
                all_gold_rank = rank
        else:
            cumulative.update(result_evidence_id_set(result))
            if gold_ids and all_gold_rank is None and gold_ids <= cumulative:
                all_gold_rank = rank

    for k in k_values:
        if keyword_groups:
            matched_at_k: set[int] = set()
            for fallback_rank, result in ranked:
                rank = int(result.get("rank") or fallback_rank)
                if rank <= k:
                    for i, group in enumerate(keyword_groups):
                        if result_hits_keyword_group(result, group):
                            matched_at_k.add(i)
            stats[f"gold_coverage@{k}"] = len(matched_at_k) / gold_count if gold_count else 0.0
            stats[f"all_gold@{k}"] = bool(gold_count and len(matched_at_k) == gold_count)
        else:
            retrieved_ids: set[str] = set()
            for fallback_rank, result in ranked:
                rank = int(result.get("rank") or fallback_rank)
                if rank <= k:
                    retrieved_ids.update(result_evidence_id_set(result))
            retrieved_gold = gold_ids & retrieved_ids
            stats[f"gold_coverage@{k}"] = len(retrieved_gold) / gold_count if gold_count else 0.0
            stats[f"all_gold@{k}"] = bool(gold_ids and gold_ids <= retrieved_ids)

    stats["all_gold_rank"] = all_gold_rank
    stats["all_gold_rr"] = 1.0 / all_gold_rank if all_gold_rank else 0.0
    return stats


def summarize_rows(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    """Summarize retrieval metrics for a group of detail rows."""
    n = len(rows)
    return {
        "n": n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "all_gold_mrr": sum(r["all_gold_rr"] for r in rows) / n,
        **{f"hit@{k}": sum(1 for r in rows if r[f"hit@{k}"]) / n for k in k_values},
        **{
            f"gold_coverage@{k}": sum(r[f"gold_coverage@{k}"] for r in rows) / n
            for k in k_values
        },
        **{
            f"all_gold@{k}": sum(1 for r in rows if r[f"all_gold@{k}"]) / n
            for k in k_values
        },
    }


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

        keyword_groups = gold_keyword_groups(qa)
        gold_ids = set(gold_evidence_ids(qa))
        ranks = []
        for result in run.get("results", []):
            hit = (
                result_hits_gold_keywords(result, keyword_groups)
                if keyword_groups
                else result_hits_gold(result, gold_ids)
            )
            if hit:
                ranks.append(int(result.get("rank") or 0))
        first_rank = min(ranks) if ranks else None
        row = {
            "qa_id": qa.get("id"),
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "strategy": run.get("strategy"),
            "gold_evidence_ids": sorted(gold_ids),
            "gold_keywords": keyword_groups,
            "first_hit_rank": first_rank,
        }
        for k in k_values:
            row[f"hit@{k}"] = bool(first_rank is not None and first_rank <= k)
        row["rr"] = 1.0 / first_rank if first_rank else 0.0
        row.update(all_gold_stats(run.get("results", []), gold_ids, keyword_groups, k_values))
        details.append(row)
        by_strategy[str(run.get("strategy"))].append(row)

    summary = {}
    for strategy, rows in sorted(by_strategy.items()):
        n = len(rows)
        if n == 0:
            continue
        summary[strategy] = summarize_rows(rows, k_values)

        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_type[str(r.get("qa_type") or "unknown")].append(r)
        summary[strategy]["by_type"] = {
            qtype: summarize_rows(type_rows, k_values)
            for qtype, type_rows in sorted(by_type.items())
        }

    return {"summary": summary, "details": details}
