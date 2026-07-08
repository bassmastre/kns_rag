from .text import RE_LCO_HEADER


def col_of(w: dict, col1_max: int, col2_max: int) -> str:
    """Map a word to the condition, action, or completion-time column."""
    if w["x0"] < col1_max:
        return "COND"
    if w["x0"] < col2_max:
        return "ACT"
    return "TIME"


def _center_y(w: dict) -> float:
    return (w["top"] + w["bottom"]) / 2


def _height(w: dict) -> float:
    return max(1.0, w["bottom"] - w["top"])


def _same_visual_row(row: list[dict], w: dict) -> bool:
    row_top = min(x["top"] for x in row)
    row_bottom = max(x["bottom"] for x in row)
    row_center = (row_top + row_bottom) / 2
    row_height = max(1.0, row_bottom - row_top)

    return abs(_center_y(w) - row_center) <= row_height * 0.7


def _is_subscript(prev: dict, cur: dict) -> bool:
    """좌표 기반 아래첨자 판단."""
    prev_text = str(prev.get("text", ""))
    cur_text = str(cur.get("text", ""))

    if not prev_text or not cur_text:
        return False

    # 너무 긴 일반 단어를 붙이지 않기 위한 최소 안전장치.
    if len(cur_text) > 5:
        return False

    prev_x1 = prev.get("x1", prev.get("x0", 0.0))
    cur_x0 = cur.get("x0", 0.0)
    x_gap = cur_x0 - prev_x1

    # 아래첨자는 앞 기호 바로 오른쪽에 붙어 있음.
    if x_gap < -1.0 or x_gap > 4.0:
        return False

    prev_h = _height(prev)
    cur_h = _height(cur)

    # 아래첨자는 보통 앞 토큰보다 작거나 비슷함.
    if cur_h > prev_h * 1.15:
        return False

    # 아래첨자는 앞 토큰보다 중심 y가 아래에 있음.
    if _center_y(cur) <= _center_y(prev) + prev_h * 0.10:
        return False

    # 앞 토큰은 보통 T, P, T1 같은 짧은 기호여야 함.
    # 이 조건은 문자열 '종류'가 아니라 일반 문장 단어 오결합 방지용.
    if len(prev_text) > 3:
        return False

    return True


def join_words(ws: list[dict]) -> str:
    """Join words in visual reading order.

    pdfplumber에서 아래첨자/위첨자는 top 좌표가 달라져 순서가 밀릴 수 있다.
    먼저 visual row로 묶고, row 내부 x0 순서로 정렬한 뒤,
    좌표상 아래첨자인 토큰은 앞 토큰에 `_`로 붙인다.
    """
    if not ws:
        return ""

    rows: list[list[dict]] = []

    for w in sorted(ws, key=lambda x: (x["top"], x["x0"])):
        for row in rows:
            if _same_visual_row(row, w):
                row.append(w)
                break
        else:
            rows.append([w])

    rows.sort(key=lambda row: min(w["top"] for w in row))

    lines: list[str] = []

    for row in rows:
        parts: list[str] = []
        prev_word: dict | None = None

        for w in sorted(row, key=lambda x: x["x0"]):
            text = str(w["text"])

            if prev_word is not None and parts and _is_subscript(prev_word, w):
                parts[-1] += "_" + text
            else:
                parts.append(text)

            prev_word = w

        lines.append(" ".join(parts))

    return " ".join(lines)


def build_bands(anchor_tops: list[float], bottom: float) -> list[tuple]:
    """Build vertical bands from anchor tops to the bottom boundary."""
    tops = sorted(set(anchor_tops))
    return [
        (tops[i], tops[i + 1] if i + 1 < len(tops) else bottom)
        for i in range(len(tops))
    ]


def words_in_band(ws: list[dict], top: float, bottom: float) -> list[dict]:
    """Return words whose top coordinates fall inside a vertical band."""
    return [w for w in ws if top <= w["top"] < bottom]


def find_col_header(words: list[dict]) -> tuple | None:
    """Find the table column-header row bounds."""
    anchor = [w for w in words if w["text"] == "COMPLETION"]
    if not anchor:
        return None
    top = min(w["top"] for w in anchor)
    row = [w for w in words if abs(w["top"] - top) < 6]
    return (min(w["top"] for w in row), max(w["bottom"] for w in row))


def extract_lco_and_title(words: list[dict], header_margin: int) -> tuple[str | None, str | None]:
    """Extract the LCO identifier and title from page-header words."""
    head = [w for w in words if w["top"] < header_margin]
    head_txt = join_words(head)
    ids = [x for x in RE_LCO_HEADER.findall(head_txt) if x.count(".") == 2]
    lco = ids[0] if ids else None
    title = None
    if lco:
        idx = head_txt.find(lco)
        if idx > 0 and head_txt[:idx].strip():
            title = head_txt[:idx].strip()
    return lco, title


def find_sr_heading_top(words: list[dict], top_bound: float) -> float | None:
    """Find the top of a SURVEILLANCE section heading at/below top_bound.

    'SURVEILLANCE' 단독으로 탐지하되, 같은 행 다음 토큰이 'REQUIREMENTS'(섹션
    제목) 또는 'FREQUENCY'(SR 표 컬럼 헤더)인 경우만 앵커로 인정해 오탐을 막는다.
    섹션 제목이 header_margin 위로 흡수돼 'SURVEILLANCE FREQUENCY'만 남는 페이지
    (예: 3.4.20-2)도 잡기 위함. SR 섹션은 코퍼스에서 제외 확정이므로, 이 top이
    ACTIONS 표/narrative 하단 컷이 되어 raw·struct 양 경로에 동일 적용된다.
    """
    candidates = [w for w in words if w["top"] >= top_bound and w["text"] == "SURVEILLANCE"]
    for w in sorted(candidates, key=lambda w: w["top"]):
        row = sorted(
            (x for x in words if abs(x["top"] - w["top"]) < 6),
            key=lambda x: x["x0"],
        )
        idx = row.index(w)
        if idx + 1 < len(row) and row[idx + 1]["text"] in ("REQUIREMENTS", "FREQUENCY"):
            return w["top"]
    return None


def page_regions(words: list[dict], page, cfg: dict) -> dict:
    """Split page words into narrative and table regions."""
    layout_cfg = cfg.get("layout", cfg)
    defaults = layout_cfg.get("defaults", layout_cfg)
    header_margin = defaults["header_margin"]
    head = [w for w in words if w["top"] < header_margin]
    hdr_bot = max((w["bottom"] for w in head), default=0.0)
    foot_tops = []
    for w in words:
        t = w["text"]
        parts = t.split("-")
        is_lco_footer = (
            len(parts) == 2
            and parts[1].isdigit()
            and len(parts[0].split(".")) == 3
            and all(p.isdigit() for p in parts[0].split("."))
        )
        if (
            t == "Westinghouse STS"
            or (t.startswith("BWR/") and t[4:].isdigit())
            or t == "CE"
            or is_lco_footer
            or t.startswith("Rev.")
        ):
            foot_tops.append(w["top"])
    foot_top = min(foot_tops) if foot_tops else page.height
    # SURVEILLANCE 섹션은 코퍼스 제외 확정 — 있으면 그 top이 하단 컷.
    # COMPLETION 컬럼 헤더가 없는 SR 전용 페이지(narr로 흘러 raw_text에 누출되던
    # 경로)도 잡도록 col header 유무와 무관하게 계산해 두 branch에 동일 적용한다.
    sr_top = find_sr_heading_top(words, hdr_bot)
    bottom = sr_top if sr_top is not None else foot_top
    ch = find_col_header(words)
    if ch:
        return {"narr": (hdr_bot, ch[0]), "tbl": (ch[1], bottom)}
    return {"narr": (hdr_bot, bottom), "tbl": None}
