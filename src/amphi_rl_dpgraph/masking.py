import re
from dataclasses import dataclass
from typing import Literal

ResolvedPolicy = Literal["raw", "weak", "pseudo", "redact", "synthetic"]

# Text patterns
RE_PATIENT_FULL = re.compile(r"(Patient[:\s]+)([A-Z][a-z]+\s[A-Z][a-z]+)")
RE_PATIENT_FIRST = re.compile(r"(Patient[:\s]+)([A-Z][a-z]+)\b")

# DOB: accept both colon and space separator ("DOB: ...", "DOB ...")
RE_DOB = re.compile(r"(DOB[:\s]\s*)(\d{2}/\d{2}/\d{4})", re.IGNORECASE)

# ISO date: 2045-07-04 — PHI_PATTERN catches these regardless of context
RE_DATE_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

# Bare slash date: 01/01/1970 appearing without a DOB prefix.
# Runs AFTER RE_DOB so it catches any remaining unredacted dates.
RE_DATE_BARE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\b")


RE_MRN_DIGITS = re.compile(r"(MRN[-\s]*)(\d{7,10})", re.IGNORECASE)
RE_MRN_ALPHA  = re.compile(r"\b(MRN)([A-Z0-9]{4,})\b", re.IGNORECASE)

# Keep the old name as an alias so any external callers don't break.
RE_MRN = RE_MRN_DIGITS

# Matches both "at <Facility>" and "to <Facility>" so synthetic stream text

RE_FACILITY = re.compile(r"((?:at|to)\s*)([A-Za-z.\s]+(?:Hospital|Clinic|Center))")

# ASR patterns (looser)
RE_ASR_MRN = re.compile(r"(?:\bm\s*r\s*n\b|\bmrn\b)\s*([0-9 ]{6,12})", re.IGNORECASE)
RE_ASR_DOB = re.compile(r"(?:\bdate of birth\b|\bdob\b)\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# TEXT MODALITY
def mask_text_redact(text: str) -> str:
    t = str(text)
    t = RE_PATIENT_FULL.sub(r"\1[REDACTED]", t)
    t = RE_PATIENT_FIRST.sub(r"\1[REDACTED]", t)
    t = RE_DOB.sub(r"\1[REDACTED]", t)          # "DOB: ..." and "DOB ..."
    t = RE_MRN_DIGITS.sub(r"\1[REDACTED]", t)   # "MRN 1234567"
    t = RE_MRN_ALPHA.sub(r"\1[REDACTED]", t)    # "MRN0000"
    t = RE_DATE_ISO.sub("[REDACTED]", t)         # "2045-07-04"
    t = RE_DATE_BARE.sub("[REDACTED]", t)        # bare "01/01/1970" not caught above
    t = RE_FACILITY.sub(r"\1[REDACTED]", t)
    return _normalize_spaces(t)


def mask_text_weak(text: str) -> str:
    t = str(text)
    t = RE_DOB.sub(r"\1[REDACTED]", t)    # "DOB: ..." and "DOB ..."
    t = RE_DATE_ISO.sub("[REDACTED]", t)  # ISO dates
    t = RE_DATE_BARE.sub("[REDACTED]", t) # bare slash dates
    return _normalize_spaces(t)


def mask_text_pseudo(text: str, patient_token: str) -> str:
    """
    Stable pseudonymisation. patient_token includes version suffix so output
    changes after localized retokenization: PATIENT_123_V0 -> PATIENT_123_V1.
    """
    t = str(text)
    t = RE_PATIENT_FULL.sub(lambda m: f"{m.group(1)}{patient_token}", t)
    t = RE_PATIENT_FIRST.sub(lambda m: f"{m.group(1)}{patient_token}", t)
    t = RE_DOB.sub(r"\1DATE_TOKEN", t)                             # "DOB: ..." and "DOB ..."
    t = RE_MRN_DIGITS.sub(lambda m: f"{m.group(1)}ID_{patient_token}", t)
    t = RE_MRN_ALPHA.sub(lambda m: f"{m.group(1)}ID_{patient_token}", t)  # "MRN0000"
    t = RE_DATE_ISO.sub("DATE_TOKEN", t)                           # ISO dates
    t = RE_DATE_BARE.sub("DATE_TOKEN", t)                          # bare slash dates
    # Strip "Patient:" label so PHI_PATTERN cannot fire on the token itself.
    t = re.sub(r"Patient:\s*", "Subject: ", t)
    return _normalize_spaces(t)


# ASR MODALITY
def mask_asr_redact(text: str) -> str:
    t = str(text)
    t = re.sub(r"\bpatient\s+[a-z]+\b", "patient [REDACTED]", t, flags=re.IGNORECASE)
    t = RE_ASR_DOB.sub("date of birth [REDACTED]", t)
    t = RE_ASR_MRN.sub("mrn [REDACTED]", t)
    return _normalize_spaces(t)


def mask_asr_weak(text: str) -> str:
    t = str(text)
    t = RE_ASR_DOB.sub("date of birth [REDACTED]", t)
    return _normalize_spaces(t)


def mask_asr_pseudo(text: str, patient_token: str) -> str:
    t = str(text)
    t = re.sub(r"\bpatient\s+[a-z]+\b", f"patient {patient_token}", t, flags=re.IGNORECASE)
    t = RE_ASR_DOB.sub("date of birth DATE_TOKEN", t)
    # Trailing space prevents token merging with next word in MRN match group.
    t = RE_ASR_MRN.sub(f"mrn ID_{patient_token} ", t)
    return _normalize_spaces(t)


# Image / waveform / audio proxy leak flags
# "synthetic" behaves identically to "pseudo" for integer proxy modalities -
# the real synthetic transformation is handled in masking_ops.py / cmo_media.py
# for actual array payloads.
def image_leak_flag(raw_has_phi: int, policy: ResolvedPolicy) -> int:
    if policy == "raw":
        return int(raw_has_phi)
    if policy in {"pseudo", "redact", "synthetic"}:
        return 0
    if policy == "weak":
        return int(raw_has_phi)
    raise ValueError(f"Unknown policy for image_leak_flag: {policy!r}")


def waveform_leak_flag(raw_has_phi: int, policy: ResolvedPolicy) -> int:
    if policy == "raw":
        return int(raw_has_phi)
    if policy in {"weak", "pseudo", "redact", "synthetic"}:
        return 0
    raise ValueError(f"Unknown policy for waveform_leak_flag: {policy!r}")


def audio_leak_flag(raw_has_phi: int, policy: ResolvedPolicy) -> int:
    if policy == "raw":
        return int(raw_has_phi)
    if policy in {"pseudo", "redact", "synthetic"}:
        return 0
    if policy == "weak":
        return int(raw_has_phi)
    raise ValueError(f"Unknown policy for audio_leak_flag: {policy!r}")


@dataclass
class PolicyOutputs:
    raw: str
    redact: str
    pseudo: str
    weak: str
    synthetic: str
    adaptive: str
