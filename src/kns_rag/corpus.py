from collections import OrderedDict

from .text import extract_note, parse_label, strip_note


def _fmt_set(labels: list[str]) -> str:
    """멤버 집합 표기 {a, b}. 집합이므로 라벨 중복은 순서 보존 dedupe."""
    return "{" + ", ".join(dict.fromkeys(labels)) + "}"


def _one_connector(conns: list[str | None]) -> str | None:
    """connector 리스트에서 단일 값 도출(원문 값 사용, 그룹 번호로 추론하지 않음).

    같은 관계 내 connector는 균일한 게 정상(예: OR 대안들). 균일하면 그 값,
    비어 있으면 None, 드물게 섞이면 '/'로 결합해 드러낸다.
    """
    uniq = list(dict.fromkeys(c for c in conns if c))
    if not uniq:
        return None
    return uniq[0] if len(uniq) == 1 else "/".join(uniq)


def _as_sentence(text: str | None) -> str:
    """Append a period only when the text has no terminal punctuation."""
    if not text:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    return text if text.endswith((".", "?", "!")) else f"{text}."


def _action_text(a: dict) -> str:
    prefix = "[Optional] " if a.get("optional") else ""
    return f"{prefix}{a['label']} {a['text']}".strip()


def _action_detail_lines(a: dict) -> list[str]:
    lines = [_action_text(a)]
    if a.get("note"):
        lines.append(f"NOTE: {a['note']}")
    if a.get("completion_time"):
        lines.append(f"Completion Time: {a['completion_time']}")
    if a.get("refs"):
        lines.extend(f"See {r}." for r in a["refs"])
    return lines


def _condition_inter_clauses(
    groups: "OrderedDict[object, list[dict]]",
    order: list[object],
    own_group: object,
) -> list[str]:
    """Return adjacent inter-group connector clauses without collapsing all groups.

    각 비-첫 group의 첫 action connector는 바로 이전 group과 현재 group 사이의
    원문 connector로 취급한다. group이 3개 이상이면 모든 인접 group boundary를
    별도 절로 보존한다. 대상 action의 own group이 해당 boundary에 포함되면 own
    group을 앞에 둔다. own group과 무관한 boundary는 원래 순서로 둔다.
    """
    if len(order) <= 1:
        return []

    own_idx = order.index(own_group)
    boundaries = []
    for right_idx in range(1, len(order)):
        left_idx = right_idx - 1
        right_group = order[right_idx]
        conn = _one_connector([groups[right_group][0].get("connector")])
        if not conn:
            continue
        distance = min(abs(own_idx - left_idx), abs(own_idx - right_idx))
        boundaries.append((distance, left_idx, right_idx, conn))

    boundaries.sort(key=lambda x: (x[0], x[1]))

    clauses = []
    for _, left_idx, right_idx, conn in boundaries:
        left_group = order[left_idx]
        right_group = order[right_idx]

        if own_group == left_group:
            first_group, second_group = left_group, right_group
        elif own_group == right_group:
            first_group, second_group = right_group, left_group
        else:
            first_group, second_group = left_group, right_group

        first = _fmt_set([m["label"] for m in groups[first_group]])
        second = _fmt_set([m["label"] for m in groups[second_group]])
        clauses.append(f"the group {first} is joined with {second} by {conn}")

    return clauses


def condition_logic(actions: list[dict]) -> dict[int, str | None]:
    """한 condition의 액션별 2중 중첩 group logic 문구를 생성.

    - group 번호(원문 파싱값)로 묶어 같은 group=intra, 다른 group=inter.
    - connector는 원문 connector 값을 읽어서 사용(그룹 번호로 추론 금지).
    - 자기 group을 항상 앞에 두고 서술. 형제/다른 group이 없으면 해당 절 생략,
      둘 다 없으면(단일 액션 등) None.
    - group이 3개 이상이면 inter 관계를 하나로 뭉개지 않고 인접 group boundary별로
      connector를 보존한다.
    - 반환 키는 action id(라벨 중복 오파싱에도 청크별 유일).
    """
    groups: "OrderedDict[object, list[dict]]" = OrderedDict()
    for a in actions:
        groups.setdefault(a["group"], []).append(a)
    order = list(groups.keys())

    out: dict[int, str | None] = {}
    for a in actions:
        g = a["group"]
        members = groups[g]
        clauses = []

        siblings = [m["label"] for m in members if m is not a]
        intra_c = _one_connector([m.get("connector") for m in members[1:]])
        if siblings and intra_c:
            sib_str = siblings[0] if len(siblings) == 1 else _fmt_set(siblings)
            clauses.append(f"{a['label']} is joined with {sib_str} by {intra_c}")

        clauses.extend(_condition_inter_clauses(groups, order, g))

        out[a["id"]] = ("Logic: " + "; ".join(clauses) + ".") if clauses else None
    return out


def assemble_sections(pages: list[dict]) -> list[dict]:
    """Merge consecutive page-level raw data by LCO."""
    sections = {}
    order = []
    for pd in pages:
        lco = pd["lco"]
        if lco is None:
            continue
        if lco not in sections:
            sections[lco] = {
                "lco": lco,
                "title": pd["title"],
                "lco_stmt": None,
                "applicability": None,
                "cond_bands": [],
                "act_items": [],
                "raw_pages": [],
                "pages": [],
                "source_doc": pd.get("source_doc"),
            }
            order.append(lco)
        s = sections[lco]
        s["pages"].append(pd["page_no"])
        if pd["lco_stmt"] and not s["lco_stmt"]:
            s["lco_stmt"] = pd["lco_stmt"]
        if pd["applicability"] and not s["applicability"]:
            s["applicability"] = pd["applicability"]
        if pd["title"] and not s["title"]:
            s["title"] = pd["title"]
        if pd.get("source_doc") and not s.get("source_doc"):
            s["source_doc"] = pd["source_doc"]
        s["cond_bands"].extend(pd["cond_bands"])
        s["act_items"].extend(pd["act_items"])
        if pd.get("raw_text"):
            s["raw_pages"].append(pd["raw_text"])
    return [sections[l] for l in order]


def build_condition_chunks(
    section: dict,
    condition_blocks: list[dict],
    *,
    source_doc: str | None,
) -> list[dict]:
    """Build condition-level chunks for condition-aware retrieval/generation."""
    lco = section["lco"]
    chunks = []

    for cb in condition_blocks:
        evidence_ids = [a["gid"] for a in cb["actions"]]
        lines = [
            _as_sentence(f"LCO {lco} {section['title'] or ''}".strip()),
            _as_sentence(f"Applicability: {section['applicability']}".strip()),
            "",
            _as_sentence(f"Condition {cb['label']}: {cb['text']}".strip()),
            "",
            "Required Actions:",
        ]

        for a in cb["actions"]:
            if a.get("connector"):
                lines.extend(["", a["connector"], ""])
            lines.extend(_action_detail_lines(a))

        body = "\n".join(line for line in lines if line is not None).strip()

        chunks.append(
            {
                "id": cb["gid"],
                "metadata": {
                    "source_hierarchical_id": lco,
                    "lco": lco,
                    "title": section["title"],
                    "applicability": section["applicability"],
                    "condition_id": cb["id"],
                    "condition_label": cb["label"],
                    "chunk_type": "condition",
                    "evidence_ids": evidence_ids,
                    "source_doc": source_doc,
                    "source_pages": section["pages"],
                },
                "content": {
                    "condition_text": cb["text"],
                    "actions": cb["actions"],
                    "body": body,
                },
            }
        )

    if section["lco_stmt"]:
        lco_note = extract_note(section["lco_stmt"])
        lco_stmt_clean = strip_note(section["lco_stmt"])
        lines = [
            _as_sentence(f"LCO {lco} {section['title'] or ''}".strip()),
            _as_sentence(f"Applicability: {section['applicability']}".strip()),
            _as_sentence(lco_stmt_clean),
        ]
        if lco_note:
            lines.append(f"NOTE: {lco_note}")
        chunks.append(
            {
                "id": f"{lco}/LCO",
                "metadata": {
                    "source_hierarchical_id": lco,
                    "lco": lco,
                    "title": section["title"],
                    "applicability": section["applicability"],
                    "condition_id": None,
                    "condition_label": None,
                    "chunk_type": "lco_statement",
                    "evidence_ids": [f"{lco}/LCO"],
                    "source_doc": source_doc,
                    "source_pages": section["pages"],
                },
                "content": {
                    "condition_text": None,
                    "actions": [],
                    "body": "\n".join(line for line in lines if line).strip(),
                },
            }
        )

    return chunks


def build_records(section: dict) -> tuple[dict, list[dict], list[dict]]:
    """Build one section record, action-level source records, and condition chunks."""
    lco = section["lco"]
    cond_map = {c["letter"]: c for c in section["cond_bands"]}
    source_doc = section.get("source_doc")

    prev, actions_seq = None, []
    pending_conn = None
    for it in section["act_items"]:
        if it["type"] == "connector":
            pending_conn = it["text"]
            actions_seq.append({"type": "connector", "text": it["text"]})
            continue
        info = parse_label(it["label"])
        conn = (
            pending_conn
            if pending_conn and prev is not None and info["condition"] == prev["condition"]
            else None
        )
        actions_seq.append({"type": "action", **it, **info, "connector": conn})
        pending_conn = None
        prev = info

    grouped = {}
    cur = None
    for it in actions_seq:
        if it["type"] == "action":
            cur = it["condition"]
        if cur is not None:
            grouped.setdefault(cur, []).append(it)

    lines = []
    if section["title"]:
        lines.append(section["title"])
    if section["lco_stmt"]:
        lines.append(section["lco_stmt"])
    if section["applicability"]:
        lines.append(f"APPLICABILITY: {section['applicability']}")
    lines.append("")
    for letter, items in grouped.items():
        c = cond_map.get(letter, {"text": "", "optional": False})
        lines.append(f"CONDITION {letter}: {c['text']}")
        for it in items:
            if it["type"] == "connector":
                lines.append(f"  {it['text']}")
            else:
                note = f"(NOTE: {it['note']}) " if it["note"] else ""
                ct = f"  [CT] {it['ct']}" if it["ct"] else ""
                lines.append(f"  {it['label']} {note}{it['text']}{ct}")
        lines.append("")
    actions_text = "\n".join(lines).rstrip()

    condition_blocks = []
    for ci, (letter, items) in enumerate(grouped.items(), 1):
        c = cond_map.get(letter, {"text": "", "optional": False, "compound": False})
        acts = []
        for ai, it in enumerate([x for x in items if x["type"] == "action"], 1):
            acts.append(
                {
                    "id": ai,
                    "gid": f"{lco}/{it['label']}",
                    "label": it["label"],
                    "group": it["group"],
                    "alt": it["alt"],
                    "connector": it["connector"],
                    "optional": it["optional"],
                    "text": it["text"],
                    "completion_time": it["ct"] or None,
                    "note": it["note"],
                    "refs": it["refs"],
                }
            )
        condition_blocks.append(
            {
                "id": ci,
                "gid": f"{lco}/{letter}",
                "label": letter,
                "text": c["text"],
                "optional": c["optional"],
                "compound": c.get("compound", False),
                "actions": acts,
            }
        )

    hier = {
        "id": lco,
        "metadata": {
            "lco": lco,
            "title": section["title"],
            "applicability": section["applicability"],
            "source_doc": source_doc,
            "source_pages": section["pages"],
        },
        "content": {
            "lco_statement": section["lco_stmt"],
            "raw_text": " ".join(p for p in section.get("raw_pages", []) if p).strip()
            or None,
            "actions_text": actions_text,
            "condition_blocks": condition_blocks,
        },
    }

    flats = []
    for cb in condition_blocks:
        logic_map = condition_logic(cb["actions"])
        for a in cb["actions"]:
            ref_str = (" " + " ".join(f"See {r}." for r in a["refs"])) if a["refs"] else ""
            note_str = f" Note: {a['note']}." if a["note"] else ""
            parts = [
                _as_sentence(f"LCO {lco} {section['title'] or ''}".strip()),
                _as_sentence(f"Applicability: {section['applicability']}".strip()),
                _as_sentence(f"Condition {cb['label']}: {cb['text']}".strip()),
                f"Required Action {a['label']}: {a['text']}"
                f"{note_str}"
                f" Completion Time: {a['completion_time']}.{ref_str}",
            ]
            body = " ".join(p for p in parts if p).strip()
            logic = logic_map.get(a["id"])
            if logic:
                body = f"{body} {logic}"
            flats.append(
                {
                    "id": a["gid"],
                    "metadata": {
                        "source_hierarchical_id": lco,
                        "lco": lco,
                        "title": section["title"],
                        "applicability": section["applicability"],
                        "condition_id": cb["id"],
                        "condition_label": cb["label"],
                        "action_id": a["id"],
                        "action_label": a["label"],
                        "connector": a["connector"],
                        "optional": a["optional"],
                        "source_doc": source_doc,
                        "source_pages": section["pages"],
                    },
                    "content": {
                        "condition_text": cb["text"],
                        "action_text": a["text"],
                        "completion_time": a["completion_time"],
                        "note": a["note"],
                        "refs": a["refs"],
                        "logic": logic,
                        "body": body,
                    },
                }
            )
    if section["lco_stmt"]:
        lco_note = extract_note(section["lco_stmt"])
        lco_stmt_clean = strip_note(section["lco_stmt"])
        note_str = f" Note: {lco_note}." if lco_note else ""
        body = " ".join(
            p
            for p in [
                _as_sentence(f"LCO {lco} {section['title'] or ''}".strip()),
                _as_sentence(f"Applicability: {section['applicability']}".strip()),
                f"{lco_stmt_clean}{note_str}".strip(),
            ]
            if p
        ).strip()
        flats.append(
            {
                "id": f"{lco}/LCO",
                "metadata": {
                    "source_hierarchical_id": lco,
                    "lco": lco,
                    "title": section["title"],
                    "applicability": section["applicability"],
                    "condition_id": None,
                    "condition_label": None,
                    "action_id": None,
                    "action_label": None,
                    "chunk_type": "lco_statement",
                    "source_doc": source_doc,
                    "source_pages": section["pages"],
                },
                "content": {
                    "condition_text": None,
                    "action_text": None,
                    "completion_time": None,
                    "note": lco_note,
                    "refs": [],
                    "logic": None,
                    "body": body,
                },
            }
        )

    condition_chunks = build_condition_chunks(
        section,
        condition_blocks,
        source_doc=source_doc,
    )
    return hier, flats, condition_chunks
