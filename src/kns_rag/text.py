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
    """Normalize extracted text while preserving operational notes.

    \ub300\uc2dc \uc2a4\uce90\ud3f4\ub529(NOTE/NOTES/REVIEWER'S NOTE)\uc744 normalize_raw\uc640 \ub3d9\uc77c \uaddc\uce59\uc73c\ub85c
    \ucc98\ub9ac\ud55c\ub2e4. REVIEWER'S NOTE\ub294 \ud3b8\uc9d1\uc6a9 \uc8fc\uc11d\uc774\ubbc0\ub85c \ubcf8\ubb38\uc9f8 \uc81c\uac70, \uc77c\ubc18 NOTE/NOTES\ub294
    \ub300\uc2dc\ub9cc \ubc97\uae30\uace0 \ubcf8\ubb38\uc744 (NOTE: ...) \ub9c8\ucee4\ub85c \uac10\uc2fc\ub2e4 \u2014 action/LCO note \ubd84\ub9ac
    (extract_note/strip_note)\uac00 \uc774 \ub9c8\ucee4\uc5d0 \uc758\uc874\ud55c\ub2e4. \uc9dd \uc5c6\ub294 \ud5e4\ub354\u00b7\uace0\ub9bd \ub300\uc2dc \ub7f0\ub3c4 \uc81c\uac70.
    \ub2e8\uc218\u00b7\ubcf5\uc218(NOTE/NOTES)\ub97c \ubaa8\ub450 \ucc98\ub9ac\ud55c\ub2e4.
    """
    # 1) REVIEWER'S NOTE \ud3b8\uc9d1 \ube14\ub85d: \ubcf8\ubb38\uc9f8 \uc81c\uac70 (\ub2e8\uc218\u00b7\ubcf5\uc218)
    txt = re.sub(
        r"-{3,}\s*REVIEWER['\u2019]S\s+NOTES?\s*-{3,}.*?-{3,}",
        " ",
        txt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 2) \uc77c\ubc18 NOTE/NOTES \uc2a4\uce90\ud3f4\ub529: \ub300\uc2dc\ub9cc \ubc97\uae30\uace0 \ubcf8\ubb38\uc744 (NOTE: ...)\ub85c \uac10\uc308
    txt = re.sub(
        r"-{3,}\s*NOTES?\s*-{3,}(.*?)-{3,}",
        lambda m: f"(NOTE: {re.sub(r'\s+', ' ', m.group(1)).strip()})",
        txt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 3) \uc9dd \uc5c6\ub294(\ub2eb\ub294 \ub300\uc2dc \uc5c6\ub294) NOTE/NOTES \ud5e4\ub354 \uc81c\uac70
    txt = re.sub(r"-{3,}\s*NOTES?\s*-{3,}", " ", txt, flags=re.IGNORECASE)
    # 4) \ub0a8\uc740 \uace0\ub9bd \ub300\uc2dc \ub7f0 \uc81c\uac70
    txt = re.sub(r"-{3,}", " ", txt)
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


def normalize_raw(txt: str) -> str:
    """Minimal normalization for raw document text.

    대시 스캐폴딩(----NOTE----, ----NOTES----, ----REVIEWER'S NOTE----)만
    제거하고 노트 본문은 평문으로 남긴다. 인쇄된 실제 토큰(A., A.1, AND, OR,
    대괄호)은 보존. 구조 마커는 주입하지 않는다 — baseline 청킹 입력용.
    """
    # 1) REVIEWER'S NOTE 편집 블록: 본문째 제거
    txt = re.sub(
        r"-{3,}\s*REVIEWER['\u2019]S\s+NOTES?\s*-{3,}.*?-{3,}",
        " ",
        txt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 2) 일반 ----NOTE---- / ----NOTES---- 스캐폴딩: 대시만 벗기고 본문 유지
    txt = re.sub(
        r"-{3,}\s*NOTES?\s*-{3,}(.*?)-{3,}",
        lambda m: " " + m.group(1).strip() + " ",
        txt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # 3) 짝 없는(닫는 대시 없는) NOTE 헤더 스캐폴딩 제거
    txt = re.sub(r"-{3,}\s*NOTES?\s*-{3,}", " ", txt, flags=re.IGNORECASE)
    # 4) 남은 고립 대시 런 제거
    txt = re.sub(r"-{3,}", " ", txt)
    # 5) 대괄호(plant-specific 마커) 제거 — 안쪽 본문은 유지("[or repair]"->"or
    #    repair"). struct(clean_text)와 동일 처리로 검색 단위 텍스트를 일치시킨다.
    #    optional 정보는 struct optional 필드에 이미 보존되어 정보 손실 없음.
    txt = txt.replace("[", "").replace("]", "")
    return re.sub(r"\s+", " ", txt).strip()