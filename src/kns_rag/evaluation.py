from __future__ import annotations

from collections import defaultdict
from typing import Any


def gold_keyword_groups(qa: dict[str, Any]) -> list[list[str]]:
    """Return gold keyword groups where groups are OR and keywords inside a group are AND."""
    value = qa["gold_keywords"]
    groups = []
    for item in value:
        group = [str(x) for x in item if str(x).strip()]
        if group:
            groups.append(group)
    return groups


def gold_evidence_ids(qa: dict[str, Any]) -> list[str]:
    """Return strict gold evidence ids for the ID-based debug path."""
    return [str(x) for x in qa["gold_evidence_ids"] if str(x).strip()]


def normalize_match_text(text: str) -> str:
    """Normalize text for case-insensitive keyword containment checks."""
    return " ".join(str(text).lower().split())


def result_hits_keyword_group(result: dict[str, Any], keyword_group: list[str]) -> bool:
    """Return whether a retrieved result contains every keyword in one group."""
    if not keyword_group:
        return False
    body = normalize_match_text(result.get("body") or "")
    if not body:
        return False
    return all(normalize_match_text(keyword) in body for keyword in keyword_group)


def result_hits_gold_keywords(result: dict[str, Any], keyword_groups: list[list[str]]) -> bool:
    """Return whether a retrieved result contains any full gold keyword group."""
    return any(result_hits_keyword_group(result, group) for group in keyword_groups)


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


def result_hits_gold(result: dict[str, Any], gold_ids: set[str]) -> bool:
    """Return whether a retrieval result overlaps any gold evidence id."""
    if not gold_ids:
        return False
    return bool(result_evidence_id_set(result) & gold_ids)


def ranked_results(results: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Return retrieval results sorted by rank with fallback ranks."""
    return sorted(
        (
            (int(result.get("rank") or fallback_rank), result)
            for fallback_rank, result in enumerate(results, 1)
        ),
        key=lambda pair: pair[0],
    )


def set_recall_stats(results: list[dict[str, Any]], keyword_groups: list[list[str]], k_values: list[int]) -> dict[str, Any]:
    """Compute keyword-group recall and set-recall metrics in one ranked pass."""
    gold_count = len(keyword_groups)
    stats: dict[str, Any] = {"gold_count": gold_count}
    thresholds = sorted(set(k_values))
    snapshots: dict[int, int] = {}
    matched_groups: set[int] = set()
    set_recall_rank = None
    threshold_idx = 0

    for rank, result in ranked_results(results):
        while threshold_idx < len(thresholds) and thresholds[threshold_idx] < rank:
            snapshots[thresholds[threshold_idx]] = len(matched_groups)
            threshold_idx += 1
        for i, group in enumerate(keyword_groups):
            if result_hits_keyword_group(result, group):
                matched_groups.add(i)
        if set_recall_rank is None and gold_count and len(matched_groups) == gold_count:
            set_recall_rank = rank
        while threshold_idx < len(thresholds) and thresholds[threshold_idx] == rank:
            snapshots[thresholds[threshold_idx]] = len(matched_groups)
            threshold_idx += 1

    while threshold_idx < len(thresholds):
        snapshots[thresholds[threshold_idx]] = len(matched_groups)
        threshold_idx += 1

    for k in k_values:
        matched_count = snapshots.get(k, 0)
        stats[f"recall@{k}"] = matched_count / gold_count if gold_count else 0.0
        stats[f"set_recall@{k}"] = bool(gold_count and matched_count == gold_count)
    stats["set_recall_rank"] = set_recall_rank
    stats["set_recall_rr"] = 1.0 / set_recall_rank if set_recall_rank else 0.0
    return stats


def set_recall_stats_by_ids(results: list[dict[str, Any]], gold_ids: set[str], k_values: list[int]) -> dict[str, Any]:
    """Compute ID-based recall and set-recall metrics in one ranked pass."""
    gold_count = len(gold_ids)
    stats: dict[str, Any] = {"gold_count": gold_count}
    thresholds = sorted(set(k_values))
    snapshots: dict[int, int] = {}
    retrieved_ids: set[str] = set()
    set_recall_rank = None
    threshold_idx = 0

    for rank, result in ranked_results(results):
        while threshold_idx < len(thresholds) and thresholds[threshold_idx] < rank:
            snapshots[thresholds[threshold_idx]] = len(gold_ids & retrieved_ids)
            threshold_idx += 1
        retrieved_ids.update(result_evidence_id_set(result))
        if set_recall_rank is None and gold_ids and gold_ids <= retrieved_ids:
            set_recall_rank = rank
        while threshold_idx < len(thresholds) and thresholds[threshold_idx] == rank:
            snapshots[thresholds[threshold_idx]] = len(gold_ids & retrieved_ids)
            threshold_idx += 1

    while threshold_idx < len(thresholds):
        snapshots[thresholds[threshold_idx]] = len(gold_ids & retrieved_ids)
        threshold_idx += 1

    for k in k_values:
        matched_count = snapshots.get(k, 0)
        stats[f"recall@{k}"] = matched_count / gold_count if gold_count else 0.0
        stats[f"set_recall@{k}"] = bool(gold_ids and matched_count == gold_count)
    stats["set_recall_rank"] = set_recall_rank
    stats["set_recall_rr"] = 1.0 / set_recall_rank if set_recall_rank else 0.0
    return stats


def summarize_rows(rows: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    """Summarize retrieval metrics for a group of detail rows."""
    n = len(rows)
    return {
        "n": n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "set_recall_mrr": sum(r["set_recall_rr"] for r in rows) / n,
        **{f"hit@{k}": sum(1 for r in rows if r[f"hit@{k}"]) / n for k in k_values},
        **{f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in k_values},
        **{f"set_recall@{k}": sum(1 for r in rows if r[f"set_recall@{k}"]) / n for k in k_values},
    }


def evaluate_run_records(
    qa_records: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
    *,
    k_values: list[int],
) -> dict[str, Any]:
    """Evaluate retrieval runs using gold_keywords containment."""
    qa_by_id = {q.get("id"): q for q in qa_records}
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    for run in run_records:
        qa = qa_by_id.get(run.get("qa_id"))
        if qa is None:
            continue

        keyword_groups = gold_keyword_groups(qa)
        ranks = []
        for result in run.get("results", []):
            if result_hits_gold_keywords(result, keyword_groups):
                ranks.append(int(result.get("rank") or 0))
        first_rank = min(ranks) if ranks else None
        row = {
            "qa_id": qa.get("id"),
            "qa_type": qa.get("type") or qa.get("qa_type"),
            "strategy": run.get("strategy"),
            "gold_keywords": keyword_groups,
            "first_hit_rank": first_rank,
        }
        for k in k_values:
            row[f"hit@{k}"] = bool(first_rank is not None and first_rank <= k)
        row["rr"] = 1.0 / first_rank if first_rank else 0.0
        row.update(set_recall_stats(run.get("results", []), keyword_groups, k_values))
        details.append(row)
        by_strategy[str(run.get("strategy"))].append(row)

    summary = {}
    for strategy, rows in sorted(by_strategy.items()):
        if not rows:
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


def evaluate_run_records_by_ids(
    qa_records: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
    *,
    k_values: list[int],
) -> dict[str, Any]:
    """Evaluate retrieval runs using strict gold_evidence_ids matching."""
    qa_by_id = {q.get("id"): q for q in qa_records}
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []

    for run in run_records:
        qa = qa_by_id.get(run.get("qa_id"))
        if qa is None:
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
        row.update(set_recall_stats_by_ids(run.get("results", []), gold_ids, k_values))
        details.append(row)
        by_strategy[str(run.get("strategy"))].append(row)

    summary = {}
    for strategy, rows in sorted(by_strategy.items()):
        if not rows:
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
