# Unified apply_masking() dispatch for all modalities and policy tiers.
# Text and ASR are handled by string transforms in masking.py; image/waveform/audio
# by array or flag operations in cmo_media.py. Proxy-integer payloads return
# a binary phi-present flag instead of a transformed array. Synthetic policy maps
# to apply_synthetic_replacement() for text/ASR, and to blur/pitch-shift for arrays.

from __future__ import annotations

from typing import Any, Literal, Union

from .masking import (
    mask_text_redact, mask_text_weak, mask_text_pseudo,
    mask_asr_redact, mask_asr_weak, mask_asr_pseudo,
    image_leak_flag, waveform_leak_flag, audio_leak_flag,
)
from .cmo_media import (
    apply_gaussian_blur, redact_image_overlay, image_phi_flag,
    obfuscate_voice, mute_audio_segment, audio_phi_flag,
    mask_waveform_header, waveform_phi_flag,
    apply_synthetic_replacement,
)

Modality       = Literal["text", "asr", "image_proxy", "waveform_proxy", "audio_proxy"]
ResolvedPolicy = Literal["raw", "weak", "pseudo", "redact", "synthetic"]


def _is_proxy_int(payload: Any) -> bool:
    return isinstance(payload, (int, float)) or (
        isinstance(payload, str) and payload.isdigit()
    )


def apply_masking(
    *,
    modality: Modality,
    policy: ResolvedPolicy,
    payload: Any,
    patient_token: str = "PATIENT_0_V0",
) -> Union[str, int, Any]:
    if modality == "text":
        s = str(payload)
        if policy == "raw":      return s
        if policy == "weak":     return mask_text_weak(s)
        if policy == "synthetic": return apply_synthetic_replacement(s)
        if policy == "pseudo":   return mask_text_pseudo(s, patient_token)
        if policy == "redact":   return mask_text_redact(s)
        raise ValueError(f"Unknown policy: {policy}")

    if modality == "asr":
        s = str(payload)
        if policy == "raw":      return s
        if policy == "weak":     return mask_asr_weak(s)
        if policy == "synthetic": return apply_synthetic_replacement(s)
        if policy == "pseudo":   return mask_asr_pseudo(s, patient_token)
        if policy == "redact":   return mask_asr_redact(s)
        raise ValueError(f"Unknown policy: {policy}")

    if modality == "image_proxy":
        if _is_proxy_int(payload):
            return (
                image_phi_flag(payload, policy)
                if policy != "synthetic"
                else image_phi_flag(payload, "pseudo")
            )
        if policy == "raw":    return payload
        if policy == "weak":   return apply_gaussian_blur(payload, kernel_size=9)
        if policy in ("pseudo", "synthetic"): return apply_gaussian_blur(payload, kernel_size=21)
        if policy == "redact": return redact_image_overlay(payload)
        raise ValueError(f"Unknown policy: {policy}")

    if modality == "waveform_proxy":
        if _is_proxy_int(payload):
            return waveform_phi_flag(payload, policy)
        if policy == "raw":
            return payload
        return mask_waveform_header(payload, patient_token)

    if modality == "audio_proxy":
        if _is_proxy_int(payload):
            return (
                audio_phi_flag(payload, policy)
                if policy != "synthetic"
                else audio_phi_flag(payload, "pseudo")
            )
        if policy == "raw":    return payload
        if policy == "weak":   return obfuscate_voice(payload, method="pitch_shift")
        if policy in ("pseudo", "synthetic"): return obfuscate_voice(payload, method="pitch_shift")
        if policy == "redact": return mute_audio_segment(payload)
        raise ValueError(f"Unknown policy: {policy}")

    raise ValueError(f"Unknown modality: {modality}")
