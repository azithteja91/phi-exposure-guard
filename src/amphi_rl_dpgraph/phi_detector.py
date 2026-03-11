import re
from typing import FrozenSet, List, Tuple

_SYNTHETIC_FIRST: FrozenSet[str] = frozenset({
    "Alex", "Blake", "Casey", "Dana", "Elliot", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kendall", "Lane", "Morgan", "Noel", "Oakley", "Parker",
    "Quinn", "Reese", "Sage", "Taylor", "Uri", "Vale", "Wren", "Xen", "Yael", "Zara",
})

_SYNTHETIC_LAST: FrozenSet[str] = frozenset({
    "Avery", "Brooks", "Chen", "Dalton", "Ellis", "Flynn", "Grant", "Hayes",
    "Irving", "James", "Kent", "Lowe", "Mason", "Nash", "Owen", "Price",
    "Quinn", "Reed", "Shaw", "Todd", "Upton", "Vane", "Ward", "Yates",
})

_SYNTHETIC_FULL_NAMES: FrozenSet[str] = frozenset(
    f"{f} {l}" for f in _SYNTHETIC_FIRST for l in _SYNTHETIC_LAST
)

_SYNTHETIC_ANY_NAME: FrozenSet[str] = _SYNTHETIC_FIRST | _SYNTHETIC_LAST

_SYNTHETIC_DATES: FrozenSet[str] = frozenset({
    "01/15/2045", "03/22/2047", "07/04/2051", "11/11/2053", "02/28/2049",
    "06/17/2055", "09/09/2043", "12/31/2057", "04/01/2061", "08/19/2063",
    "05/05/2067", "10/30/2069", "01/01/2071", "03/14/2073", "07/07/2075",
    "11/23/2041", "02/02/2039", "06/06/2037", "09/19/2035", "12/12/2033",
})

PATIENT_TOKEN_PREFIX = r"PATIENT_"
_PSEUDO_PREFIXES = rf"(?:{PATIENT_TOKEN_PREFIX}|ID_{PATIENT_TOKEN_PREFIX})"

_NAME = r"[a-z]{2,}"

_pattern_str = (
    r"("
    r"\b\d{7,10}\b"
    r"|\b\d{2}/\d{2}/\d{4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b"
    r"|Patient:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
    + r"|\bpatient\s+(?!" + _PSEUDO_PREFIXES + r")" + _NAME + r"(?:\s+" + _NAME + r")?\b"
    + r"|\bmrn\b\s*(?!ID_" + PATIENT_TOKEN_PREFIX + r")[0-9 ]{6,12}\b"
    + r"|\bMRN\d{4,}\b"
    + r")"
)
PHI_PATTERN = re.compile(_pattern_str, re.IGNORECASE)

def _synthetic_mrn(original_mrn: str) -> str:
    """Mirror of cmo_media.synthetic_mrn — kept local to avoid circular import."""
    digits = re.sub(r"\D", "", str(original_mrn))
    if not digits:
        return original_mrn
    result = []
    for i, d in enumerate(digits):
        seed = f"{original_mrn}:{i}"
        h = 0
        for c in seed:
            h = (h * 31 + ord(c)) & 0xFFFFFFFF
        shift = (h % 9) + 1
        result.append(str((int(d) + shift) % 10))
    return "".join(result)

def _is_synthetic_match(match_text: str) -> bool:
    """
    Return True if the matched text is a synthetic replacement rather than real PHI.
    """
    t = match_text.strip()

    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", t) and t in _SYNTHETIC_DATES:
        return True

    digits_only = re.sub(r"[^0-9]", "", t)
    if len(digits_only) in range(7, 11):
        if re.fullmatch(r"\d{7,10}", digits_only):
            stripped = re.sub(r"(?i)^mrn\s*", "", t).strip()
            if re.fullmatch(r"\d{7,10}", stripped):
                return True

    t_clean = re.sub(r"^patient[:\s]+", "", t, flags=re.IGNORECASE).strip()
    t_clean = re.sub(r"\s+mrn\s*$", "", t_clean, flags=re.IGNORECASE).strip()

    parts = t_clean.split()

    if len(parts) >= 2:
        two = f"{parts[0]} {parts[1]}".title()
        if two in _SYNTHETIC_FULL_NAMES:
            return True

    if len(parts) >= 1 and parts[0].title() in _SYNTHETIC_ANY_NAME:
        return True

    return False

def find_phi_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if text is None:
        return spans
    s = str(text)
    for m in PHI_PATTERN.finditer(s):
        if not _is_synthetic_match(m.group(0)):
            spans.append((m.start(), m.end()))
    return spans

def count_phi(text: str) -> int:
    if text is None:
        return 0
    return sum(
        1 for m in PHI_PATTERN.finditer(str(text))
        if not _is_synthetic_match(m.group(0))
    )

def leakage(text: str) -> int:
    return count_phi(text)

def avg_leaks_per_note(texts) -> float:
    texts = list(texts)
    if not texts:
        return 0.0
    return sum(count_phi(t) for t in texts) / len(texts)
