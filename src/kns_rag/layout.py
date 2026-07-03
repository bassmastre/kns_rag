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
    ch = find_col_header(words)
    if ch:
        return {"narr": (hdr_bot, ch[0]), "tbl": (ch[1], foot_top)}
    return {"narr": (hdr_bot, foot_top), "tbl": None}
