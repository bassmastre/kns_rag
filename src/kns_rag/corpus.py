from .text import extract_note, parse_label, strip_note


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


def build_records(section: dict) -> tuple[dict, list[dict]]:
    """Build one hierarchical record and its flat records from a section."""
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
        for a in cb["actions"]:
            ref_str = (" " + " ".join(f"See {r}." for r in a["refs"])) if a["refs"] else ""
            note_str = f" Note: {a['note']}." if a["note"] else ""
            body = (
                f"LCO {lco} {section['title'] or ''}. "
                f"Applicability: {section['applicability']}. "
                f"Condition {cb['label']}: {cb['text']} "
                f"Required Action {a['label']}: {a['text']}"
                f"{note_str}"
                f" Completion Time: {a['completion_time']}.{ref_str}"
            ).strip()
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
                        "body": body,
                    },
                }
            )
    if section["lco_stmt"]:
        lco_note = extract_note(section["lco_stmt"])
        lco_stmt_clean = strip_note(section["lco_stmt"])
        note_str = f" Note: {lco_note}." if lco_note else ""
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
                    "body": (
                        f"LCO {lco} {section['title'] or ''}. "
                        f"Applicability: {section['applicability']}. "
                        f"{lco_stmt_clean}"
                        f"{note_str}"
                    ).strip(),
                },
            }
        )

    return hier, flats
