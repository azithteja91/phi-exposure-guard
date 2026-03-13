# ExposurePolicyController wires ContextState and DCPGAdapter into a single
# decision surface. record_and_decide records the event, computes risk
# components, primes the embedding cache, runs cross-modal similarity checks,
# applies a link-bonus nudge when co-references are detected, optionally
# relaxes policy when utility_delta breaches the floor, triggers localized
# remask on threshold crossings, and attaches a DCPG snapshot when remask fires.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_state import ContextState, RiskComponents
from .dcpg import DCPGAdapter


@dataclass(frozen=True)
class Decision:
    policy_name: str
    reason: str

    risk_pre: float
    risk_post: Optional[float]
    risk_source: str

    localized_remask: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    candidates: List[str] = field(default_factory=list)

    link_modalities_recent: List[str] = field(default_factory=list)
    risk_components: Dict[str, Any] = field(default_factory=dict)

    cross_modal_matches: List[str] = field(default_factory=list)

    relaxed_for_utility: bool = False
    override_reason: Optional[str] = None


class ExposurePolicyController:
    def __init__(
        self,
        *,
        context: ContextState,
        risk_1: float = 0.40,
        risk_2: float = 0.80,
        remask_thresh: float = 0.75,
        utility_floor: float = -0.10,
        cross_modal_sim_threshold: float = 0.30,
    ):
        self.context   = context
        self.risk_1    = float(risk_1)
        self.risk_2    = float(risk_2)
        self.risk_mid  = (self.risk_1 + self.risk_2) / 2.0
        self.remask_thresh            = float(remask_thresh)
        self.utility_floor            = float(utility_floor)
        self.cross_modal_sim_threshold = float(cross_modal_sim_threshold)

        self._graph_adapter: Optional[DCPGAdapter] = None

    def _adapter(self) -> DCPGAdapter:
        if self._graph_adapter is None:
            self._graph_adapter = DCPGAdapter(self.context)
        return self._graph_adapter

    def current_token(self, patient_key: str, patient_id: int) -> str:
        v = int(self.context.get_pseudonym_version(str(patient_key)))
        return f"PATIENT_{int(patient_id)}_V{v}"

    def _risk_components_dict(self, comps: RiskComponents) -> Dict[str, Any]:
        return {
            "effective_units":  int(comps.effective_units),
            "units_factor":     float(comps.units_factor),
            "recency_factor":   float(comps.recency_factor),
            "link_bonus":       float(comps.link_bonus),
            "risk":             float(comps.risk),
            "provisional_risk": float(comps.provisional_risk),
            "degree":           int(comps.degree),
            "confidence":       float(comps.confidence),
        }

    def update_context_and_score(
        self,
        *,
        patient_key: str,
        event_id: str,
        timestamp: float,
        modality_exposures: Dict[str, int],
        link_signals: Optional[Dict[str, int]] = None,
    ) -> RiskComponents:
        self.context.record_event(
            patient_key=str(patient_key),
            event_id=str(event_id),
            ts=float(timestamp),
            modality_exposures=modality_exposures,
            link_signals=link_signals or {},
        )
        return self.context.risk_components(str(patient_key), now_ts=float(timestamp))

    def decide_from_risk(self, risk_pre: float, utility_delta: float = 0.0) -> Decision:
        _tiers = ["weak", "synthetic", "pseudo", "redact"]

        if risk_pre < self.risk_1:      pol, reason = "weak",      "low risk"
        elif risk_pre < self.risk_mid:  pol, reason = "synthetic", "medium-low risk: synthetic PHI replacement"
        elif risk_pre < self.risk_2:    pol, reason = "pseudo",    "medium-high risk: pseudonymization"
        else:                           pol, reason = "redact",    "high risk"

        relaxed         = False
        override_reason = None
        if utility_delta < self.utility_floor and risk_pre < self.risk_2 and pol != "weak":
            idx             = _tiers.index(pol)
            pol             = _tiers[idx - 1]
            reason          = reason + " (relaxed: utility floor breached)"
            relaxed         = True
            override_reason = "utility_floor_relaxation"

        return Decision(
            policy_name=pol,
            reason=reason,
            risk_pre=float(risk_pre),
            risk_post=None,
            risk_source="exposure_entropy_plus_link_bonus",
            localized_remask={},
            thresholds={
                "risk_1":        self.risk_1,
                "risk_mid":      self.risk_mid,
                "risk_2":        self.risk_2,
                "remask_thresh": self.remask_thresh,
                "utility_floor": self.utility_floor,
                "synthetic_band": (self.risk_1, self.risk_mid),
            },
            candidates=["weak", "synthetic", "pseudo", "redact"],
            relaxed_for_utility=relaxed,
            override_reason=override_reason,
        )

    def record_and_decide(
        self,
        *,
        patient_key: str,
        event_id: str,
        timestamp: float,
        modality_exposures: Dict[str, int],
        link_signals: Optional[Dict[str, int]] = None,
        event_payloads: Optional[Dict[str, Any]] = None,
        utility_delta: float = 0.0,
    ) -> Decision:
        pk            = str(patient_key)
        eid           = str(event_id)
        ts            = float(timestamp)
        event_payloads = dict(event_payloads or {})

        comps    = self.update_context_and_score(
            patient_key=pk,
            event_id=eid,
            timestamp=ts,
            modality_exposures=modality_exposures,
            link_signals=link_signals or {},
        )
        risk_pre  = float(comps.risk)
        link_mods = self.context.link_modalities_recent(pk)

        for mod, payload in event_payloads.items():
            if payload is not None:
                try:
                    self._adapter().invalidate_embedding(pk, mod)
                    self._adapter()._get_embedding(pk, mod, payload)
                except Exception:
                    pass

        cross_modal_matches: List[str] = []
        for mod, payload in event_payloads.items():
            if payload is None:
                continue
            try:
                matches = self._adapter().cross_modal_match(
                    pk,
                    mod,
                    payload,
                    threshold=self.cross_modal_sim_threshold,
                )
                cross_modal_matches.extend(
                    m for m in matches if m not in cross_modal_matches
                )
            except Exception:
                pass

        risk_source = "exposure_entropy_plus_link_bonus"
        if cross_modal_matches and risk_pre < self.risk_2:
            risk_pre    = min(1.0, risk_pre + 0.15)
            risk_source = "exposure_entropy_plus_cross_modal_similarity"

        risk_without_bonus = max(
            0.0,
            min(1.0, 0.8 * float(comps.units_factor) + 0.2 * float(comps.recency_factor)),
        )
        link_was_decisive = (
            float(comps.link_bonus) > 0.0
            and risk_without_bonus < self.remask_thresh
            and float(comps.risk) >= self.remask_thresh
        )
        if link_was_decisive or float(comps.link_bonus) > 0.0:
            trigger_reason = "cross_modal_link_bonus"
        else:
            trigger_reason = "threshold_crossing"

        remask = self.context.localized_remask_trigger(
            pk,
            event_id=eid,
            ts=ts,
            threshold=self.remask_thresh,
            lookback_events=10,
            trigger_reason=trigger_reason,
        )

        if remask.get("trigger"):
            try:
                snap = self._adapter().graph_summary(pk)
                remask["dcpg_snapshot_at_trigger"] = snap
            except Exception:
                pass

        decision = self.decide_from_risk(risk_pre, utility_delta=utility_delta)
        return Decision(
            policy_name=decision.policy_name,
            reason=decision.reason,
            risk_pre=decision.risk_pre,
            risk_post=decision.risk_post,
            risk_source=risk_source,
            localized_remask=remask,
            thresholds=decision.thresholds,
            candidates=decision.candidates,
            link_modalities_recent=list(link_mods),
            risk_components=self._risk_components_dict(comps),
            cross_modal_matches=cross_modal_matches,
            relaxed_for_utility=decision.relaxed_for_utility,
            override_reason=decision.override_reason,
        )

    def apply_post_masking_credit(self, patient_key: str, masked_units: int) -> Dict[str, Any]:
        return self.context.record_masking_credit(
            patient_key=str(patient_key),
            masked_units=int(masked_units),
        )
