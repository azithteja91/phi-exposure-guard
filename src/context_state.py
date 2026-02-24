
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RiskComponents:
    effective_units: int
    units_factor: float
    recency_factor: float
    link_bonus: float
    risk: float
    provisional_risk: float = 0.0
    degree: int = 0
    confidence: float = 1.0


class ContextState:
    """SQLite-backed patient exposure state and rolling risk model."""

    def __init__(
        self,
        db_path: str,
        *,
        k_units: float = 0.05,
        recency_half_life_s: float = 120.0,
        link_lookback_events: int = 20,  # increased so cross-modal links remain visible in snapshot
        link_bonus_two_modal: float = 0.20,
        link_bonus_three_modal: float = 0.30,
        provisional_k: float = 0.30,
    ) -> None:
        self.db_path = str(db_path)
        self.k_units = float(k_units)
        self.recency_half_life_s = float(recency_half_life_s)
        self.link_lookback_events = int(link_lookback_events)
        self.link_bonus_two_modal = float(link_bonus_two_modal)
        self.link_bonus_three_modal = float(link_bonus_three_modal)
        self.provisional_k = float(provisional_k)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def __enter__(self) -> "ContextState":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None  # do not suppress exceptions

    def __del__(self) -> None:
        # Belt-and-suspenders fallback — context manager is preferred.
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patient_state (
              patient_key TEXT PRIMARY KEY,
              total_phi_units INTEGER NOT NULL DEFAULT 0,
              masked_credit_units INTEGER NOT NULL DEFAULT 0,
              link_signals_seen INTEGER NOT NULL DEFAULT 0,
              pseudonym_version INTEGER NOT NULL DEFAULT 0,
              last_event_ts REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS exposures (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              patient_key TEXT NOT NULL,
              event_id TEXT NOT NULL,
              ts REAL NOT NULL,
              modality TEXT NOT NULL,
              phi_units INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_exposures_pk_ts ON exposures(patient_key, ts);
            CREATE INDEX IF NOT EXISTS idx_exposures_pk_event ON exposures(patient_key, event_id);

            CREATE TABLE IF NOT EXISTS last_risk (
              patient_key TEXT PRIMARY KEY,
              last_risk REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS remask_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              patient_key TEXT NOT NULL,
              event_id TEXT NOT NULL,
              ts REAL NOT NULL,
              old_version INTEGER NOT NULL,
              new_version INTEGER NOT NULL,
              risk_at_trigger REAL NOT NULL,
              affected_event_ids_json TEXT,
              trigger_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_remask_pk_ts ON remask_events(patient_key, ts);
            """
        )
        self._conn.commit()

    def _ensure_patient(self, patient_key: str) -> None:
        self._conn.execute(
            """
            INSERT INTO patient_state(patient_key)
            VALUES (?)
            ON CONFLICT(patient_key) DO NOTHING
            """,
            (patient_key,),
        )
        self._conn.execute(
            """
            INSERT INTO last_risk(patient_key, last_risk)
            VALUES (?, 0.0)
            ON CONFLICT(patient_key) DO NOTHING
            """,
            (patient_key,),
        )

    def get_pseudonym_version(self, patient_key: str) -> int:
        self._ensure_patient(patient_key)
        row = self._conn.execute(
            "SELECT pseudonym_version FROM patient_state WHERE patient_key=?",
            (patient_key,),
        ).fetchone()
        return int(row["pseudonym_version"]) if row else 0

    def record_masking_credit(self, *, patient_key: str, masked_units: int) -> Dict[str, Any]:
        """Masking credit reduces effective exposure, lowering future risk."""
        pk = str(patient_key)
        self._ensure_patient(pk)
        mu = max(0, int(masked_units))
        with self._conn:
            self._conn.execute(
                """
                UPDATE patient_state
                SET masked_credit_units = masked_credit_units + ?
                WHERE patient_key=?
                """,
                (mu, pk),
            )
        row = self._conn.execute(
            "SELECT total_phi_units, masked_credit_units FROM patient_state WHERE patient_key=?",
            (pk,),
        ).fetchone()
        total = int(row["total_phi_units"]) if row else 0
        credited = int(row["masked_credit_units"]) if row else 0
        return {
            "patient_key": pk,
            "masked_units_applied": mu,
            "total_phi_units": total,
            "masked_credit_units": credited,
            "effective_units": max(0, total - credited),
        }

    def record_event(
        self,
        *,
        patient_key: str,
        event_id: str,
        ts: float,
        modality_exposures: Dict[str, int],
        link_signals: Optional[Dict[str, int]] = None,
    ) -> None:
        pk = str(patient_key)
        eid = str(event_id)
        now_ts = float(ts)
        link_signals = dict(link_signals or {})

        self._ensure_patient(pk)

        rows: List[Tuple[str, str, float, str, int]] = []
        for mod, units in modality_exposures.items():
            rows.append((pk, eid, now_ts, str(mod), int(units)))

        for link_mod, val in link_signals.items():
            if int(val) > 0:
                base = str(link_mod).removesuffix("_link")
                rows.append((pk, eid, now_ts, f"{base}_link", 1))

        add_phi_units = sum(u for (_, _, _, m, u) in rows if not m.endswith("_link"))
        add_link_units = sum(u for (_, _, _, m, u) in rows if m.endswith("_link") and u > 0)

        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO exposures(patient_key, event_id, ts, modality, phi_units)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.execute(
                """
                UPDATE patient_state
                SET total_phi_units = total_phi_units + ?,
                    link_signals_seen = link_signals_seen + ?,
                    last_event_ts = CASE WHEN last_event_ts < ? THEN ? ELSE last_event_ts END
                WHERE patient_key=?
                """,
                (int(add_phi_units), int(add_link_units), now_ts, now_ts, pk),
            )

    def _effective_units(self, pk: str) -> int:
        row = self._conn.execute(
            "SELECT total_phi_units, masked_credit_units FROM patient_state WHERE patient_key=?",
            (pk,),
        ).fetchone()
        if not row:
            return 0
        return max(0, int(row["total_phi_units"]) - int(row["masked_credit_units"]))

    def _recency_factor(self, pk: str, now_ts: Optional[float]) -> float:
        row = self._conn.execute(
            "SELECT last_event_ts FROM patient_state WHERE patient_key=?",
            (pk,),
        ).fetchone()
        if not row:
            return 0.0
        last_ts = float(row["last_event_ts"] or 0.0)
        now = time.time() if now_ts is None else float(now_ts)
        age = max(0.0, now - last_ts)
        if self.recency_half_life_s <= 0:
            return 1.0
        return float(math.pow(0.5, age / self.recency_half_life_s))

    def link_modalities_recent(self, pk: str) -> List[str]:
        rows = self._conn.execute(
            """
            SELECT modality, phi_units
            FROM exposures
            WHERE patient_key=?
            ORDER BY ts DESC, id DESC
            LIMIT ?
            """,
            (pk, int(self.link_lookback_events)),
        ).fetchall()
        mods: set[str] = set()
        for r in rows:
            m = str(r["modality"])
            u = int(r["phi_units"])
            if m.endswith("_link") and u > 0:
                mods.add(m)
        return sorted(mods)

    def _link_bonus(self, pk: str) -> float:
        n = len(self.link_modalities_recent(pk))
        if n >= 3:
            return float(self.link_bonus_three_modal)
        if n >= 2:
            return float(self.link_bonus_two_modal)
        return 0.0

    def _degree(self, pk: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT modality) AS deg FROM exposures WHERE patient_key=?",
            (pk,),
        ).fetchone()
        return int(row["deg"]) if row else 0

    def _provisional_risk(self, effective_units: int, units_factor: float, degree: int) -> float:
        confidence = float(units_factor)
        r = 1.0 - math.exp(-self.provisional_k * float(effective_units)) * confidence / (degree + 1)
        return float(max(0.0, min(1.0, r)))

    def risk_components(self, patient_key: str, now_ts: Optional[float] = None) -> RiskComponents:
        pk = str(patient_key)
        self._ensure_patient(pk)

        effective_units = int(self._effective_units(pk))
        units_factor = 1.0 - math.exp(-self.k_units * float(effective_units))
        recency = float(self._recency_factor(pk, now_ts))
        link_bonus = float(self._link_bonus(pk))
        degree = int(self._degree(pk))

        raw = (0.8 * float(units_factor)) + (0.2 * float(recency)) + float(link_bonus)
        risk = max(0.0, min(1.0, float(raw)))
        prov_risk = self._provisional_risk(effective_units, units_factor, degree)

        return RiskComponents(
            effective_units=effective_units,
            units_factor=float(units_factor),
            recency_factor=float(recency),
            link_bonus=float(link_bonus),
            risk=float(risk),
            provisional_risk=float(prov_risk),
            degree=int(degree),
            confidence=float(units_factor),
        )

    def risk_score(self, patient_key: str, now_ts: Optional[float] = None) -> float:
        return float(self.risk_components(patient_key, now_ts=now_ts).risk)

    def localized_remask_trigger(
        self,
        patient_key: str,
        *,
        event_id: str,
        ts: float,
        threshold: float,
        lookback_events: int = 10,
        trigger_reason: str = "",
    ) -> Dict[str, Any]:
        """
        Increment pseudonym version if risk crosses threshold.
        Edge-detection: fires once on crossing, not repeatedly while above threshold.
        """
        pk = str(patient_key)
        eid = str(event_id)
        now_ts = float(ts)
        self._ensure_patient(pk)

        comps = self.risk_components(pk, now_ts=now_ts)
        risk = float(comps.risk)

        prev_row = self._conn.execute(
            "SELECT last_risk FROM last_risk WHERE patient_key=?",
            (pk,),
        ).fetchone()
        prev = float(prev_row["last_risk"]) if prev_row else 0.0

        crossed = (prev < float(threshold)) and (risk >= float(threshold))

        with self._conn:
            self._conn.execute(
                "UPDATE last_risk SET last_risk=? WHERE patient_key=?",
                (float(risk), pk),
            )

        if not crossed:
            return {
                "trigger": False,
                "risk": float(risk),
                "provisional_risk": float(comps.provisional_risk),
                "threshold": float(threshold),
                "affected_event_ids": [],
                "old_version": None,
                "new_version": None,
                "trigger_reason": "",
            }

        rows = self._conn.execute(
            """
            SELECT event_id
            FROM exposures
            WHERE patient_key=?
            ORDER BY ts DESC, id DESC
            LIMIT ?
            """,
            (pk, int(max(1, lookback_events)) * 6),
        ).fetchall()

        affected: List[str] = []
        seen: set[str] = set()
        for r in rows:
            ev = str(r["event_id"])
            if ev not in seen:
                seen.add(ev)
                affected.append(ev)
            if len(affected) >= int(lookback_events):
                break

        state = self._conn.execute(
            "SELECT pseudonym_version FROM patient_state WHERE patient_key=?",
            (pk,),
        ).fetchone()
        old_v = int(state["pseudonym_version"]) if state else 0
        new_v = old_v + 1

        if not trigger_reason:
            trigger_reason = "threshold_crossing"

        with self._conn:
            self._conn.execute(
                "UPDATE patient_state SET pseudonym_version=? WHERE patient_key=?",
                (int(new_v), pk),
            )
            self._conn.execute(
                """
                INSERT INTO remask_events(
                  patient_key, event_id, ts, old_version, new_version,
                  risk_at_trigger, affected_event_ids_json, trigger_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pk,
                    eid,
                    now_ts,
                    int(old_v),
                    int(new_v),
                    float(risk),
                    json.dumps(affected),
                    str(trigger_reason),
                ),
            )

        return {
            "trigger": True,
            "risk": float(risk),
            "provisional_risk": float(comps.provisional_risk),
            "threshold": float(threshold),
            "affected_event_ids": affected,
            "old_version": int(old_v),
            "new_version": int(new_v),
            "trigger_reason": str(trigger_reason),
        }
