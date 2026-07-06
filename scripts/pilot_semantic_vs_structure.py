"""파일럿: 3.4.15 하나로 semantic 청킹 vs 구조 인식형 청킹이
Condition E의 AND(조건 트리거)/OR(대안 조치) 짝을 갈라놓는지 확인.

본실험(4전략 비교) 이전 sanity check 용도의 1회성 스크립트.
- semantic: sentence-transformers 임베딩 코사인 거리 기반 breakpoint 청킹
  (LlamaIndex SemanticSplitterNodeParser와 동일한 percentile-threshold 방식)
- structure: CONDITION 경계로만 자르는 구조 인식형 청킹 (실제 hierarchical
  전략의 summary node/2단계 검색은 없음 — 이번 파일럿은 "조건 단위 보존"
  여부만 확인)

실행:
    python scripts/pilot_semantic_vs_structure.py
"""

import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
SECTIONS_PATH = ROOT / "data" / "processed" / "sections.jsonl"
BREAKPOINT_PERCENTILE = 75  # 표준 semantic chunking breakpoint 방식


def load_lines() -> list[str]:
    with SECTIONS_PATH.open("r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    text = rec["content"]["actions_text"]
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def structure_chunks(lines: list[str]) -> list[list[str]]:
    """CONDITION 헤더 경계로만 자르는 구조 인식형 청킹."""
    chunks, cur = [], []
    for ln in lines:
        if re.match(r"^CONDITION [A-Z]:", ln) and cur:
            chunks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        chunks.append(cur)
    return chunks


def semantic_chunks(lines: list[str], model: SentenceTransformer) -> list[list[str]]:
    """임베딩 코사인 거리 breakpoint 기반 semantic 청킹."""
    embs = model.encode(lines, normalize_embeddings=True)
    sims = np.array([np.dot(embs[i], embs[i + 1]) for i in range(len(embs) - 1)])
    dist = 1 - sims
    threshold = np.percentile(dist, BREAKPOINT_PERCENTILE)
    chunks, cur = [], [lines[0]]
    for i, d in enumerate(dist):
        if d > threshold:
            chunks.append(cur)
            cur = []
        cur.append(lines[i + 1])
    if cur:
        chunks.append(cur)
    return chunks, dist, threshold


def find_chunk(chunks: list[list[str]], needle_pred) -> int | None:
    for ci, ch in enumerate(chunks):
        if any(needle_pred(ln) for ln in ch):
            return ci
    return None


def report(label: str, chunks: list[list[str]], target_conditions: list[str]):
    print(f"\n=== {label}: {len(chunks)} chunks ===")
    for ci, ch in enumerate(chunks):
        print(f"--- chunk {ci} ({len(ch)} lines) ---")
        for ln in ch:
            print(f"    {ln}")

    for letter in target_conditions:
        cond_idx = find_chunk(chunks, lambda ln, l=letter: ln.startswith(f"CONDITION {l}:"))
        action_idxs = {
            ln.split()[0]: find_chunk(chunks, lambda x, ln=ln: x == ln)
            for ch in chunks
            for ln in ch
            if re.match(rf"^{letter}\.\d", ln)
        }
        together = len(set([cond_idx, *action_idxs.values()])) == 1
        print(
            f"[{label}] Condition {letter}: cond_chunk={cond_idx}, "
            f"action_chunks={action_idxs}, all_in_one_chunk={together}"
        )


def main():
    lines = load_lines()
    print(f"total lines: {len(lines)}")

    struct = structure_chunks(lines)
    report("structure-aware", struct, ["D", "E"])

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    sem, dist, threshold = semantic_chunks(lines, model)
    print(f"\n(semantic breakpoint threshold @p{BREAKPOINT_PERCENTILE} = {threshold:.4f})")
    report("semantic", sem, ["D", "E"])


if __name__ == "__main__":
    main()
