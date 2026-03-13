# Consent token and policy resolution for per-patient masking caps. A
# ConsentToken carries the patient's maximum allowed policy, any disallowed
# modalities, and an optional expiry timestamp. resolve_policy enforces a
# fixed precedence: expired token -> redact, disallowed modality -> redact,
# policy exceeds cap -> capped, otherwise the chosen policy passes through
# unchanged with status "ok".

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Set, Tuple

_POLICY_RANK = {"raw": 0, "weak": 1, "synthetic": 2, "pseudo": 3, "redact": 4}
_RANK_POLICY = {v: k for k, v in _POLICY_RANK.items()}
POLICY_ORDER = ["raw", "weak", "synthetic", "pseudo", "redact"]

ConsentStatus = str


@dataclass(frozen=True)
class ConsentToken:
    patient_key: str
    max_policy: str
    disallowed_modalities: FrozenSet[str] = field(default_factory=frozenset)
    expires_at: Optional[float] = None
    token_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = "standard"


def is_expired(token: ConsentToken, now: Optional[float] = None) -> bool:
    if token.expires_at is None:
        return False
    return (now if now is not None else time.time()) > token.expires_at


def resolve_policy(
    chosen: str,
    token: ConsentToken,
    modality: str,
    now: Optional[float] = None,
) -> Tuple[str, ConsentStatus, Optional[str]]:
    if is_expired(token, now):
        return "redact", "expired", "expired_consent_token"

    if modality in token.disallowed_modalities:
        return "redact", "modality_denied", "modality_denied_by_consent"

    cap_rank    = _POLICY_RANK.get(token.max_policy, 4)
    chosen_rank = _POLICY_RANK.get(chosen, 4)
    if chosen_rank > cap_rank:
        return token.max_policy, "capped", "consent_cap"

    return chosen, "ok", None
