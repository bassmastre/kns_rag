import re

import pdfplumber

from .layout import (
    build_bands,
    col_of,
    extract_lco_and_title,
    join_words,
    page_regions,
    words_in_band,
)
from .text import (
    RE_ACTION_LABEL,
    RE_ACTIONS_HDR,
    RE_APPLIC,
    RE_COND_LABEL,
    RE_CONNECTOR,
    RE_SR_REF,
    RE_STRIP_ALABEL,
    RE_STRIP_CLABEL,
    clean_text,
    extract_note,
    is_optional,
    normalize_raw,
    strip_bracket,
    strip_note,
)

def parse_page_raw(page, cfg: dict) -> dict | None:
    """Parse one pdfplumber page into raw section fields."""
    words = page.extract_words()
    if not words:
        return None

    layout_cfg = cfg["layout"]
    defaults = dict(layout_cfg["defaults"])
    lco, title = extract_lco_and_title(words, defaults["header_margin"])
    page_layout = dict(defaults)
    if lco:
        page_layout.update(layout_cfg.get("overrides", {}).get(lco, {}))
    if page_layout["header_margin"] != defaults["header_margin"]:
        lco, title = extract_lco_and_title(words, page_layout["header_margin"])
    reg = page_regions(words, page, page_layout)

    n_top, n_bot = reg["narr"]
    nws = [w for w in words if n_top <= w["top"] < n_bot]
    lco_stmt, applic = None, None
    if nws:
        ntxt = RE_ACTIONS_HDR.split(clean_text(join_words(nws)))[0].strip()
        ap = RE_APPLIC.search(ntxt)
        if ap:
            stmt_raw = ntxt[: ap.start()].strip()
            applic = RE_APPLIC.sub("", ntxt[ap.start() :].strip(), count=1)
            applic = applic.lstrip(":").strip()
        else:
            stmt_raw = ntxt
            applic = None
        m = re.search(r"(LCO\s+\d+\.\d+\.\d+\s+.*)", stmt_raw, re.DOTALL)
        lco_stmt = m.group(1).strip() if m else (stmt_raw or None)

    cond_bands_raw, act_items_raw = [], []
    if reg["tbl"]:
        t_top, t_bot = reg["tbl"]
        tws = [w for w in words if t_top <= w["top"] < t_bot]
        cond_ws = [
            w
            for w in tws
            if col_of(w, page_layout["col1_max"], page_layout["col2_max"]) == "COND"
        ]
        act_ws = [
            w
            for w in tws
            if col_of(w, page_layout["col1_max"], page_layout["col2_max"]) == "ACT"
        ]
        time_ws = [
            w
            for w in tws
            if col_of(w, page_layout["col1_max"], page_layout["col2_max"]) == "TIME"
        ]

        act_sorted = sorted(act_ws, key=lambda w: (round(w["top"]), w["x0"]))
        act_anchors = []
        for i, w in enumerate(act_sorted):
            t = w["text"]
            if RE_ACTION_LABEL.match(t) or RE_CONNECTOR.match(t):
                act_anchors.append(w)
            elif (
                t == "["
                and i + 1 < len(act_sorted)
                and RE_ACTION_LABEL.match(act_sorted[i + 1]["text"])
            ):
                act_anchors.append(act_sorted[i + 1])
        cond_anchors = [w for w in cond_ws if RE_COND_LABEL.match(w["text"])]

        for top, bottom in (
            build_bands([w["top"] for w in cond_anchors], t_bot)
            if cond_anchors
            else []
        ):
            raw = join_words(words_in_band(cond_ws, top, bottom))
            m = re.match(r"^([A-Z])\.", raw)
            if m:
                cleaned = clean_text(raw)
                cond_bands_raw.append(
                    {
                        "letter": m.group(1),
                        "text": RE_STRIP_CLABEL.sub("", cleaned),
                        "optional": is_optional(raw),
                        "compound": bool(re.search(r"\bAND\b|\bOR\b", cleaned)),
                    }
                )

        for top, bottom in (
            build_bands([w["top"] for w in act_anchors], t_bot)
            if act_anchors
            else []
        ):
            bw = sorted(
                words_in_band(act_ws, top, bottom),
                key=lambda w: (round(w["top"]), w["x0"]),
            )
            if not bw:
                continue
            first = None
            for w in bw:
                cand = strip_bracket(w["text"])
                if cand:
                    first = cand
                    break
            if first is None:
                continue
            raw_act = join_words(words_in_band(act_ws, top, bottom))
            raw_ct = join_words(words_in_band(time_ws, top, bottom))
            if RE_CONNECTOR.match(first):
                act_items_raw.append({"type": "connector", "text": first})
            elif RE_ACTION_LABEL.match(first):
                cleaned = clean_text(raw_act)
                body = RE_STRIP_ALABEL.sub("", cleaned)
                act_items_raw.append(
                    {
                        "type": "action",
                        "label": first,
                        "note": extract_note(body),
                        "text": strip_note(body),
                        "ct": clean_text(raw_ct),
                        "optional": is_optional(raw_act),
                        "refs": RE_SR_REF.findall(cleaned),
                    }
                )
    return {
        "lco": lco,
        "title": title,
        "lco_stmt": lco_stmt,
        "applicability": applic,
        "cond_bands": cond_bands_raw,
        "act_items": act_items_raw,
        "raw_text": page_raw_text(words, reg, page_layout, lco),
    }


def narrative_from_lco(nws: list[dict], lco: str | None) -> list[dict]:
    """Drop a leading section-heading row ('3.4.x <title>') from the narrative.

    첫 페이지 narrative 선두엔 페이지 헤더 제목이 한 번 더 인쇄된 라인이 있어
    비교군(raw_text)만 제목을 이중으로 갖게 된다. 'LCO <id>' 문장 라인부터
    시작하도록 그 위 라인을 잘라 struct 경로(LCO ...부터 정규식 추출)와 맞춘다.
    LCO 라인이 없는 연속 페이지는 그대로 둔다.
    """
    rows: dict[int, list[dict]] = {}
    for w in nws:
        rows.setdefault(round(w["top"]), []).append(w)
    lco_top = None
    for top in sorted(rows):
        if re.match(r"^LCO\s+\d+\.\d+\.\d+\b", join_words(rows[top])):
            lco_top = top
            break
    if lco_top is None:
        return nws
    return [w for w in nws if round(w["top"]) >= lco_top]


def page_raw_text(
    words: list[dict], reg: dict, page_layout: dict, lco: str | None = None
) -> str | None:
    """Linearize one page into plain document text for baseline chunking.

    Reads columns correctly (avoids extract_text() column mixing) but injects
    no structure: only tokens physically printed in the PDF, in reading order.
    Table is read condition-band by condition-band (COND -> ACT -> TIME within
    each band) so a condition stays next to its own actions; band boundaries
    are used only to reconstruct reading order, not passed to the chunker.
    Region bounds follow page_regions (SURVEILLANCE section already excluded).
    """
    parts: list[str] = []

    n_top, n_bot = reg["narr"]
    nws = narrative_from_lco(
        [w for w in words if n_top <= w["top"] < n_bot], lco
    )
    if nws:
        parts.append(normalize_raw(join_words(nws)))

    if reg["tbl"]:
        t_top, t_bot = reg["tbl"]
        tws = [w for w in words if t_top <= w["top"] < t_bot]
        c1, c2 = page_layout["col1_max"], page_layout["col2_max"]
        cond_ws = [w for w in tws if col_of(w, c1, c2) == "COND"]
        act_ws = [w for w in tws if col_of(w, c1, c2) == "ACT"]
        time_ws = [w for w in tws if col_of(w, c1, c2) == "TIME"]

        anchors = sorted({w["top"] for w in cond_ws if RE_COND_LABEL.match(w["text"])})
        if anchors:
            for top, bottom in build_bands(anchors, t_bot):
                seg = " ".join(
                    x
                    for x in (
                        normalize_raw(join_words(words_in_band(cond_ws, top, bottom))),
                        normalize_raw(join_words(words_in_band(act_ws, top, bottom))),
                        normalize_raw(join_words(words_in_band(time_ws, top, bottom))),
                    )
                    if x
                )
                if seg:
                    parts.append(seg)
        else:
            for col_ws in (cond_ws, act_ws, time_ws):
                seg = normalize_raw(join_words(col_ws))
                if seg:
                    parts.append(seg)

    text = normalize_raw(" ".join(p for p in parts if p))
    return text or None