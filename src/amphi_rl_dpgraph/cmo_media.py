# CMO media transforms for PHI-bearing payloads across text, image, audio, and
# waveform modalities. Synthetic replacement functions use deterministic hashing
# so the same input always maps to the same output, preserving cross-modal
# co-reference. Image and audio CMOs fall back gracefully when numpy/cv2 are
# absent. All regex patterns are module-level compiled constants.

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import random

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

_RE_NAME         = re.compile(r"\b([A-Z][a-z]+)\s([A-Z][a-z]+)\b")
_RE_PATIENT_FULL = re.compile(r"((?:Patient|patient)[:\s]+)([A-Z][a-z]+\s[A-Z][a-z]+)")
_RE_PATIENT_SOLO = re.compile(r"((?:Patient|patient)[:\s]+)([A-Z][a-z]+)\b")
_RE_ASR_SOLO     = re.compile(r"(\bpatient\s+)([a-z]{2,})\b", re.IGNORECASE)
_RE_DATE         = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
_RE_DATE_ISO     = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_MRN_LABELLED = re.compile(r"(MRN\s*)(\d{7,10})", re.IGNORECASE)
_RE_MRN_ALPHA    = re.compile(r"\b(MRN)([A-Z0-9]{4,})\b", re.IGNORECASE)
_RE_MRN_BARE     = re.compile(r"\b(\d{7,10})\b")


def _deterministic_index(seed_str: str, pool_len: int) -> int:
    h = 0
    for c in seed_str:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h % pool_len


def synthetic_name(original: str) -> str:
    fi = _deterministic_index(original, len(_SYNTHETIC_FIRST))
    li = _deterministic_index(original[::-1], len(_SYNTHETIC_LAST))
    return f"{_SYNTHETIC_FIRST[fi]} {_SYNTHETIC_LAST[li]}"


_SYNTHETIC_DATES = [
    "01/15/2045", "03/22/2047", "07/04/2051", "11/11/2053", "02/28/2049",
    "06/17/2055", "09/09/2043", "12/31/2057", "04/01/2061", "08/19/2063",
    "05/05/2067", "10/30/2069", "01/01/2071", "03/14/2073", "07/07/2075",
    "11/23/2041", "02/02/2039", "06/06/2037", "09/19/2035", "12/12/2033",
]

_SYNTHETIC_DATES_SET: frozenset = frozenset(_SYNTHETIC_DATES)


def synthetic_date(date_str: str) -> str:
    idx = _deterministic_index(str(date_str), len(_SYNTHETIC_DATES))
    return _SYNTHETIC_DATES[idx]


def synthetic_mrn(original_mrn: str) -> str:
    digits = re.sub(r"\D", "", str(original_mrn))
    if not digits:
        return original_mrn
    result = []
    for i, d in enumerate(digits):
        seed  = f"{original_mrn}:{i}"
        shift = (_deterministic_index(seed, 9) + 1)
        result.append(str((int(d) + shift) % 10))
    return "".join(result)


def apply_gaussian_blur(payload: Any, kernel_size: int = 15) -> Any:
    try:
        import numpy as np
        arr = np.asarray(payload, dtype=np.uint8)
        try:
            import cv2
            return cv2.GaussianBlur(arr, (kernel_size | 1, kernel_size | 1), 0)
        except ImportError:
            pass

        from numpy.lib.stride_tricks import sliding_window_view
        k   = max(3, kernel_size | 1)
        pad = k // 2
        if arr.ndim == 3:
            out = np.zeros_like(arr)
            for c in range(arr.shape[2]):
                ch      = np.pad(arr[:, :, c], pad, mode="edge").astype(np.float32)
                windows = sliding_window_view(ch, (k, k))
                out[:, :, c] = windows.mean(axis=(-1, -2)).astype(np.uint8)
            return out
        return arr
    except Exception:
        return payload


def redact_image_overlay(payload: Any) -> Any:
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


def shift_pitch(payload: Any, shift_factor: float = 1.15) -> Any:
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


def mask_waveform_header(header: Any, patient_token: str = "PATIENT_0_V0") -> Any:
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


def replace_names_synthetic(text: str) -> str:
    t = str(text)

    def _replace_full(m: re.Match) -> str:
        return synthetic_name(m.group(0))

    t = _RE_NAME.sub(_replace_full, t)

    def _replace_patient_solo(m: re.Match) -> str:
        name = m.group(2)
        fi   = _deterministic_index(name, len(_SYNTHETIC_FIRST))
        return m.group(1) + _SYNTHETIC_FIRST[fi]

    t = _RE_PATIENT_SOLO.sub(_replace_patient_solo, t)

    def _replace_asr_solo(m: re.Match) -> str:
        name = m.group(2)
        fi   = _deterministic_index(name.lower(), len(_SYNTHETIC_FIRST))
        return m.group(1) + _SYNTHETIC_FIRST[fi].lower()

    t = _RE_ASR_SOLO.sub(_replace_asr_solo, t)

    return t


def replace_dates_synthetic(text: str) -> str:
    def _replace(m: re.Match) -> str:
        return synthetic_date(m.group(0))

    def _replace_iso(m: re.Match) -> str:
        original = f"{m.group(2)}/{m.group(3)}/{m.group(1)}"
        return synthetic_date(original)

    t = _RE_DATE.sub(_replace, str(text))
    t = _RE_DATE_ISO.sub(_replace_iso, t)
    return t


def replace_mrns_synthetic(text: str) -> str:
    t = str(text)

    def _replace_alpha(m: re.Match) -> str:
        digits = re.sub(r"[^0-9]", "", m.group(2))
        synth  = synthetic_mrn(digits) if digits else m.group(2)
        return m.group(1) + synth

    t = _RE_MRN_ALPHA.sub(_replace_alpha, t)

    def _replace_labelled(m: re.Match) -> str:
        return m.group(1) + synthetic_mrn(m.group(2))

    t = _RE_MRN_LABELLED.sub(_replace_labelled, t)

    def _replace_bare(m: re.Match) -> str:
        return synthetic_mrn(m.group(0))

    t = _RE_MRN_BARE.sub(_replace_bare, t)
    return t


def apply_synthetic_replacement(text: str) -> str:
    t = replace_names_synthetic(text)
    t = replace_mrns_synthetic(t)
    t = replace_dates_synthetic(t)
    return t
