from .text import RE_LCO_HEADER


def col_of(w: dict, col1_max: int, col2_max: int) -> str:
    """Map a word to the condition, action, or completion-time column."""
    if w["x0"] < col1_max:
        return "COND"
    if w["x0"] < col2_max:
        return "ACT"
    return "TIME"


def join_words(ws: list[dict]) -> str:
    """Join words in visual reading order."""
    ws = sorted(ws, key=lambda w: (round(w["top"]), w["x0"]))
    return " ".join(w["text"] for w in ws)


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
