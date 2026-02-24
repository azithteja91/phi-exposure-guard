from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import random

## Synthetic name pool

_SYNTHETIC_FIRST = [
    "Alex", "Blake", "Casey", "Dana", "Elliot", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kendall", "Lane", "Morgan", "Noel", "Oakley", "Parker",
    "Quinn", "Reese", "Sage", "Taylor", "Uri", "Vale", "Wren", "Xen", "Yael", "Zara",
]

_SYNTHETIC_LAST = [
    "Avery", "Brooks", "Chen", "Dalton", "Ellis", "Flynn", "Grant", "Hayes",
    "Irving", "James", "Kent", "Lowe", "Mason", "Nash", "Owen", "Price",
    "Quinn", "Reed", "Shaw", "Todd", "Upton", "Vane", "Ward", "Yates",
]

_RE_NAME = re.compile(r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b")
_RE_PATIENT_FULL = re.compile(r"((?:Patient|patient)[:\s]+)([A-Z][a-z]+\s[A-Z][a-z]+)")
_RE_PATIENT_SOLO = re.compile(r"((?:Patient|patient)[:\s]+)([A-Z][a-z]+)\b")
_RE_ASR_SOLO = re.compile(r"(\bpatient\s+)([a-z]{2,})\b", re.IGNORECASE)
_RE_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
_RE_MRN_LABELLED = re.compile(r"(MRN\s*)(\d{7,10})", re.IGNORECASE)
_RE_MRN_BARE = re.compile(r"\b(\d{7,10})\b")


def _deterministic_index(seed_str: str, pool_len: int) -> int:
    """Stable index into a pool based on a seed string."""
    h = 0
    for c in seed_str:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h % pool_len


def synthetic_name(original: str) -> str:
    """Replace a real name with a consistent synthetic one derived from the original."""
    fi = _deterministic_index(original, len(_SYNTHETIC_FIRST))
    li = _deterministic_index(original[::-1], len(_SYNTHETIC_LAST))
    return f"{_SYNTHETIC_FIRST[fi]} {_SYNTHETIC_LAST[li]}"


# Synthetic date pool

_SYNTHETIC_DATES = [
    "01/15/2045", "03/22/2047", "07/04/2051", "11/11/2053", "02/28/2049",
    "06/17/2055", "09/09/2043", "12/31/2057", "04/01/2061", "08/19/2063",
    "05/05/2067", "10/30/2069", "01/01/2071", "03/14/2073", "07/07/2075",
    "11/23/2041", "02/02/2039", "06/06/2037", "09/19/2035", "12/12/2033",
]

_SYNTHETIC_DATES_SET: frozenset = frozenset(_SYNTHETIC_DATES)


def synthetic_date(date_str: str) -> str:
    """
    Replace a real date with a deterministic synthetic date from the pool.
    Same input always produces the same synthetic date (cross-modal co-reference
    preserved). Pool dates are set in 2033-2075 - implausible as real patient
    DOBs, clearly synthetic, but structurally valid MM/DD/YYYY dates.
    """
    idx = _deterministic_index(str(date_str), len(_SYNTHETIC_DATES))
    return _SYNTHETIC_DATES[idx]


def synthetic_mrn(original_mrn: str) -> str:
    """
    Replace a real MRN with a deterministic synthetic MRN of the same digit length.
    Each digit position is independently permuted using a position-seeded hash so
    the synthetic value looks realistic (same length, all digits) but maps
    consistently from the same input - preserving cross-modal co-reference.
    """
    digits = re.sub(r"\D", "", str(original_mrn))
    if not digits:
        return original_mrn
    result = []
    for i, d in enumerate(digits):
        seed = f"{original_mrn}:{i}"
        # deterministic replacement digit - never produces the same digit as input
        # (shift by 1-9 mod 10 to guarantee difference)
        shift = (_deterministic_index(seed, 9) + 1)
        result.append(str((int(d) + shift) % 10))
    return "".join(result)


# Image CMOs

def apply_gaussian_blur(payload: Any, kernel_size: int = 15) -> Any:
    """Blur PHI-bearing image regions. Accepts numpy (H,W,C) or integer proxy."""
    try:
        import numpy as np
        arr = np.asarray(payload, dtype=np.uint8)
        try:
            import cv2
            return cv2.GaussianBlur(arr, (kernel_size | 1, kernel_size | 1), 0)
        except ImportError:
            pass

        # numpy fallback box blur
        from numpy.lib.stride_tricks import sliding_window_view
        k = max(3, kernel_size | 1)
        pad = k // 2
        if arr.ndim == 3:
            out = np.zeros_like(arr)
            for c in range(arr.shape[2]):
                ch = np.pad(arr[:, :, c], pad, mode="edge").astype(np.float32)
                windows = sliding_window_view(ch, (k, k))
                out[:, :, c] = windows.mean(axis=(-1, -2)).astype(np.uint8)
            return out
        return arr
    except Exception:
        return payload


def redact_image_overlay(payload: Any) -> Any:
    """Black-out entire image."""
    try:
        import numpy as np
        return np.zeros_like(np.asarray(payload, dtype=np.uint8))
    except Exception:
        return payload


def image_phi_flag(payload: Any, policy: str) -> int:
    raw_flag = int(payload) if isinstance(payload, (int, float)) else 1
    if policy == "raw":
        return raw_flag
    if policy in {"pseudo", "redact"}:
        return 0
    if policy == "weak":
        return raw_flag
    return raw_flag


# Audio / MFCC CMOs

def shift_pitch(payload: Any, shift_factor: float = 1.15) -> Any:
    """Scale MFCC spectral coefficients to obfuscate speaker identity."""
    try:
        import numpy as np
        arr = np.asarray(payload, dtype=np.float32)
        out = arr.copy()
        if out.ndim == 2:
            out[:, 1:] = arr[:, 1:] * float(shift_factor)
        elif out.ndim == 1:
            out[1:] = arr[1:] * float(shift_factor)
        return out
    except Exception:
        return payload


def mute_audio_segment(payload: Any) -> Any:
    try:
        import numpy as np
        return np.zeros_like(np.asarray(payload, dtype=np.float32))
    except Exception:
        return payload


def obfuscate_voice(payload: Any, method: str = "pitch_shift") -> Any:
    if method == "mute":
        return mute_audio_segment(payload)
    return shift_pitch(payload)


def audio_phi_flag(payload: Any, policy: str) -> int:
    raw_flag = int(payload) if isinstance(payload, (int, float)) else 1
    if policy == "raw":
        return raw_flag
    if policy in {"pseudo", "redact"}:
        return 0
    if policy == "weak":
        return raw_flag
    return raw_flag


# Waveform header CMOs

def mask_waveform_header(header: Any, patient_token: str = "PATIENT_0_V0") -> Any:
    """Replace PHI fields in a waveform header dict; integer proxy always returns 0."""
    if isinstance(header, dict):
        phi_fields = {
            "patient_id", "patient_name", "mrn", "dob", "date_of_birth",
            "name", "subject_id", "device_serial",
        }
        out = dict(header)
        for k in list(out.keys()):
            if k.lower() in phi_fields:
                out[k] = patient_token
        return out
    return 0


def waveform_phi_flag(payload: Any, policy: str) -> int:
    raw_flag = int(payload) if isinstance(payload, (int, float)) else 0
    return raw_flag if policy == "raw" else 0


# Synthetic replacement CMOs

def replace_names_synthetic(text: str) -> str:
    """
    Replace real names with consistent synthetic names.
    Handles three cases:
    1. Full "First Last" anywhere in text -> synthetic full name
    2. Single name after "Patient:" or "patient:" label -> synthetic first name
    3. Single lowercase name after "patient " in ASR text -> synthetic first name
    Deterministic mapping preserves cross-modal co-reference.
    """
    t = str(text)

    # 1. Full name replacement (must run before solo to avoid double-replacing)
    def _replace_full(m: re.Match) -> str:
        return synthetic_name(m.group(0))

    t = _RE_NAME.sub(_replace_full, t)

    # 2. Solo name after "Patient: " or "patient: " (text note style)
    def _replace_patient_solo(m: re.Match) -> str:
        name = m.group(2)
        fi = _deterministic_index(name, len(_SYNTHETIC_FIRST))
        return m.group(1) + _SYNTHETIC_FIRST[fi]

    t = _RE_PATIENT_SOLO.sub(_replace_patient_solo, t)

    # 3. Solo lowercase name after "patient " in ASR
    def _replace_asr_solo(m: re.Match) -> str:
        name = m.group(2)
        fi = _deterministic_index(name.lower(), len(_SYNTHETIC_FIRST))
        return m.group(1) + _SYNTHETIC_FIRST[fi].lower()

    t = _RE_ASR_SOLO.sub(_replace_asr_solo, t)

    return t


def replace_dates_synthetic(text: str) -> str:
    """
    Replace real dates with deterministic synthetic dates from the pool.
    Pool dates are set in 2033-2075 - structurally valid MM/DD/YYYY but
    implausible as real patient DOBs, making them clearly synthetic and
    exemptable by phi_detector without breaking real PHI detection.
    """
    def _replace(m: re.Match) -> str:
        return synthetic_date(m.group(0))

    return _RE_DATE.sub(_replace, str(text))


def replace_mrns_synthetic(text: str) -> str:
    """
    Replace MRN digit strings with synthetic equivalents of the same length.
    Handles both labelled form ("MRN 12345678") and bare digit runs.
    Labelled form is processed first so the label is preserved and the
    digit group is replaced; bare form mops up any remaining digit runs.
    """
    t = str(text)

    # labelled: "MRN 12345678" -> "MRN 23456789"
    def _replace_labelled(m: re.Match) -> str:
        return m.group(1) + synthetic_mrn(m.group(2))

    t = _RE_MRN_LABELLED.sub(_replace_labelled, t)

    # bare: any remaining 7-10 digit run
    def _replace_bare(m: re.Match) -> str:
        return synthetic_mrn(m.group(0))

    t = _RE_MRN_BARE.sub(_replace_bare, t)
    return t


def apply_synthetic_replacement(text: str) -> str:
    """
    Combined synthetic PHI replacement pipeline.
    Order matters: names first (before MRN so digit-only MRNs aren't confused
    with name character runs), then MRNs, then dates last (dates contain digits
    that must not be double-processed by the MRN replacer).
    """
    t = replace_names_synthetic(text)
    t = replace_mrns_synthetic(t)
    t = replace_dates_synthetic(t)
    return t
PY
