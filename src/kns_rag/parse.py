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
    RE_DASH_RUN,
    RE_NOTE_OPEN,
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


def act_note_spans(act_sorted: list[dict]) -> list[tuple[float, float]]:
    """ACT 컬럼 NOTE 대시블록의 (여는 top, 닫는 top) 구간 목록.

    여는 토큰(----NOTE----)과 다음 닫는 대시런(----) 사이. 노트 본문에 우연히
    액션 라벨꼴 토큰(예: 3.4.11 Condition C 노트의 'B.2 or E.2.')이 있어도 액션
    앵커로 오검출되지 않도록, 이 구간 '내부'를 앵커 탐지에서 배제하는 데 쓴다.
    여는 행 자체는 제외하지 않는다(실제 액션 라벨이 노트와 같은 행에 오는 경우:
    'A.1 ----NOTE----'). 닫는 대시런이 없으면 구간을 만들지 않는다.
    """
    specials = sorted(
        (
            w
            for w in act_sorted
            if RE_NOTE_OPEN.match(w["text"]) or RE_DASH_RUN.match(w["text"])
        ),
        key=lambda w: w["top"],
    )
    spans, open_top = [], None
    for w in specials:
        if RE_NOTE_OPEN.match(w["text"]):
            if open_top is None:
                open_top = w["top"]
        elif open_top is not None:
            spans.append((open_top, w["top"]))
            open_top = None
    return spans


def time_row_texts(time_words: list[dict]) -> list[str]:
    """Return cleaned TIME-column row texts in visual order."""
    rows: dict[int, list[dict]] = {}
    for w in time_words:
        rows.setdefault(round(w["top"]), []).append(w)
    return [
        txt
        for top in sorted(rows)
        if (txt := clean_text(join_words(rows[top])))
    ]


def is_completion_time_start(text: str) -> bool:
    """Strong start marker for a new Completion Time cell.

    Do not treat continuation rows such as 'OR In accordance with ...' as new
    starts. They are alternatives inside the same completion-time cell.
    """
    return bool(
        re.match(
            r"^(?:Immediately|"
            r"\d+(?:\.\d+)?\s+"
            r"(?:second|seconds|minute|minutes|hour|hours|day|days|month|months|year|years)|"
            r"Prior\s+to|Before|Upon|Once|Within)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def split_completion_time_blocks(time_words: list[dict]) -> list[str]:
    """Split a condition-band TIME column into logical CT blocks.

    The PDF may print a single CT cell across multiple visual rows, e.g.
    '1 hour' followed by '[OR In accordance with ...]'. This function keeps
    connector-prefixed rows inside the current CT block and starts a new block
    only at a strong CT start row.
    """
    blocks: list[str] = []
    current: list[str] = []

    for row in time_row_texts(time_words):
        prev = current[-1].upper() if current else None
        starts_new = is_completion_time_start(row) and current and prev not in {"AND", "OR"}
        if starts_new:
            blocks.append(clean_text(" ".join(current)))
            current = [row]
        else:
            current.append(row)

    if current:
        blocks.append(clean_text(" ".join(current)))
    return [b for b in blocks if b]


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
        note_spans = act_note_spans(act_sorted)

        def in_note(top: float) -> bool:
            return any(o < top < c for o, c in note_spans)

        def action_anchors_in_band(top: float, bottom: float) -> list[dict]:
            band_act_sorted = sorted(
                words_in_band(act_ws, top, bottom),
                key=lambda w: (round(w["top"]), w["x0"]),
            )
            anchors = []
            for i, w in enumerate(band_act_sorted):
                if in_note(w["top"]):
                    continue
                t = w["text"]
                if RE_ACTION_LABEL.match(t) or RE_CONNECTOR.match(t):
                    anchors.append(w)
                elif (
                    t == "["
                    and i + 1 < len(band_act_sorted)
                    and RE_ACTION_LABEL.match(band_act_sorted[i + 1]["text"])
                ):
                    anchors.append(band_act_sorted[i + 1])
            return anchors

        def parse_act_items_in_band(
            top: float,
            bottom: float,
            *,
            map_condition_ct: bool,
        ) -> list[dict]:
            anchors = action_anchors_in_band(top, bottom)
            if not anchors:
                return []

            items: list[dict] = []
            for a_top, a_bottom in build_bands([w["top"] for w in anchors], bottom):
                bw = sorted(
                    words_in_band(act_ws, a_top, a_bottom),
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

                raw_act = join_words(words_in_band(act_ws, a_top, a_bottom))
                raw_ct = join_words(words_in_band(time_ws, a_top, a_bottom))
                if RE_CONNECTOR.match(first):
                    items.append({"type": "connector", "text": first})
                elif RE_ACTION_LABEL.match(first):
                    cleaned = clean_text(raw_act)
                    body = RE_STRIP_ALABEL.sub("", cleaned)
                    items.append(
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

            if map_condition_ct:
                ct_blocks = split_completion_time_blocks(words_in_band(time_ws, top, bottom))
                actions = [it for it in items if it["type"] == "action"]
                if len(ct_blocks) == len(actions):
                    for action, ct in zip(actions, ct_blocks, strict=False):
                        action["ct"] = ct
                elif len(actions) == 1 and ct_blocks:
                    actions[0]["ct"] = clean_text(" ".join(ct_blocks))

            return items

        cond_anchors = [w for w in cond_ws if RE_COND_LABEL.match(w["text"])]
        cond_band_bounds = (
            build_bands([w["top"] for w in cond_anchors], t_bot) if cond_anchors else []
        )

        for top, bottom in cond_band_bounds:
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

        if cond_band_bounds:
            for top, bottom in cond_band_bounds:
                act_items_raw.extend(
                    parse_act_items_in_band(top, bottom, map_condition_ct=True)
                )
        else:
            act_items_raw.extend(
                parse_act_items_in_band(t_top, t_bot, map_condition_ct=False)
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
