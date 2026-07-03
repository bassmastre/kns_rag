import re


RE_ACTION_LABEL = re.compile(r"^[A-Z]\.\d+(?:\.\d+)?$")
RE_CONNECTOR = re.compile(r"^(?:AND|OR)$")
RE_COND_LABEL = re.compile(r"^[A-Z]\.$")
RE_APPLIC = re.compile(r"APPLICABILITY", re.IGNORECASE)
RE_ACTIONS_HDR = re.compile(r"\bACTIONS\b")
RE_LCO_HEADER = re.compile(r"\b(\d\.\d+\.\d+)\b")
RE_STRIP_ALABEL = re.compile(r"^[A-Z]\.\d+(?:\.\d+)?\s*")
RE_STRIP_CLABEL = re.compile(r"^[A-Z]\.\s*")
RE_SR_REF = re.compile(r"SR\s+\d+\.\d+\.\d+\.\d+")


def clean_text(txt: str) -> str:
    """Normalize extracted text while preserving operational notes."""
    txt = re.sub(
        r"-{3,}\s*REVIEWER['\u2019]S\s+NOTE.*?-{3,}",
        "",
        txt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    txt = re.sub(
        r"-{3,}\s*NOTE\s*-{3,}(.*?)-{3,}",
        lambda m: f"(NOTE: {re.sub(r'\s+', ' ', m.group(1)).strip()})",
        txt,
        flags=re.DOTALL,
    )
    txt = txt.replace("[", "").replace("]", "")
    return re.sub(r"\s+", " ", txt).strip()


def extract_note(txt: str) -> str | None:
    """Extract an operational note body from cleaned text."""
    m = re.search(r"\(NOTE:\s*(.*?)\)", txt)
    return m.group(1).strip() if m else None


def strip_note(txt: str) -> str:
    """Remove an operational note from text."""
    return re.sub(r"\(NOTE:.*?\)\s*", "", txt).strip()


def strip_bracket(t: str) -> str:
    """Strip a leading optional-action bracket token."""
    return t.lstrip("[").strip()


def is_optional(raw: str) -> bool:
    """Return whether raw text contains optional-action brackets."""
    return ("[" in raw) or ("]" in raw)


def parse_label(label: str) -> dict | None:
    """Parse an action label into condition, group, and alternative parts."""
    m = re.match(r"^([A-Z])\.(\d+)(?:\.(\d+))?$", label)
    if not m:
        return None
    return {
        "condition": m.group(1),
        "group": int(m.group(2)) if m.group(2) else None,
        "alt": int(m.group(3)) if m.group(3) else None,
    }


def connector_between(prev: dict | None, curr: dict | None) -> str | None:
    """Infer the connector between two parsed labels for reference only."""
    if prev is None or curr is None or prev["condition"] != curr["condition"]:
        return None
    return "OR" if prev["group"] == curr["group"] else "AND"
