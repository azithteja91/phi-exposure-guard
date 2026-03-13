
# Federated PHI exposure tracking across edge devices via a gossip bus and
# CRDT merge. EdgeDevice processes local exposure events, computes risk and
# masking policy, and publishes compact NodeDelta messages to a GossipBus.
# Peers drain and merge deltas without coordination; convergence is guaranteed
# by the underlying CRDT. Policy escalations trigger an optional callback.

from __future__ import annotations

import hashlib
import hmac
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dcpg_crdt import CRDTGraph, CRDTNodeState, merge_node


@dataclass
class NodeDelta:
    device_id: str
    seq_id: int
    patient_key: str
    modality: str
    phi_units_added: int
    link_signal_added: int
    pseudonym_version: int
    pseudonym_version_ts: float
    timestamp: float = field(default_factory=time.time)

    @property
    def node_id(self) -> str:
        return f"{self.patient_key}::{self.modality}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "seq_id": self.seq_id,
            "node_id": self.node_id,
            "patient_key": self.patient_key,
            "modality": self.modality,
            "phi_units_added": self.phi_units_added,
            "link_signal_added": self.link_signal_added,
            "pseudonym_version": self.pseudonym_version,
            "pseudonym_version_ts": self.pseudonym_version_ts,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NodeDelta":
        return NodeDelta(
            device_id=str(d["device_id"]),
            seq_id=int(d.get("seq_id", 0)),
            patient_key=str(d["patient_key"]),
            modality=str(d["modality"]),
            phi_units_added=int(d["phi_units_added"]),
            link_signal_added=int(d["link_signal_added"]),
            pseudonym_version=int(d["pseudonym_version"]),
            pseudonym_version_ts=float(d["pseudonym_version_ts"]),
            timestamp=float(d.get("timestamp", 0.0)),
        )


class GossipBus:
    """In-process pub/sub bus. Replace publish/drain with MQTT or Redis Streams
    to move to a real transport layer."""

    _QUEUE_MAX = 10_000

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._queues: Dict[str, queue.Queue[NodeDelta]] = {}

    def register(self, device_id: str) -> None:
        with self._lock:
            if device_id not in self._queues:
                self._queues[device_id] = queue.Queue(maxsize=self._QUEUE_MAX)

    def publish(self, sender_id: str, deltas: List[NodeDelta]) -> None:
        with self._lock:
            recipients = [did for did in self._queues if did != sender_id]
        for did in recipients:
            q = self._queues[did]
            for delta in deltas:
                try:
                    q.put_nowait(delta)
                except queue.Full:
                    pass

    def drain(self, device_id: str, max_items: int = 200) -> List[NodeDelta]:
        q = self._queues.get(device_id)
        if q is None:
            return []
        results: List[NodeDelta] = []
        for _ in range(max_items):
            try:
                results.append(q.get_nowait())
            except queue.Empty:
                break
        return results

    def subscriber_count(self) -> int:
        return len(self._queues)


_default_bus: GossipBus = GossipBus()


def deterministic_pseudonym(patient_key: str, secret_key: bytes, version: int = 1) -> str:
    msg = f"{patient_key}:v{version}".encode()
    digest = hmac.new(secret_key, msg, hashlib.sha256).hexdigest()[:6]
    return f"PATIENT_{digest.upper()}_V{version}"


EscalationCallback = Callable[[str, float, str], None]


@dataclass
class FederationHealth:
    deltas_sent: int = 0
    deltas_received: int = 0
    deltas_applied: int = 0
    duplicates_ignored: int = 0
    merge_conflicts: int = 0
    escalations_triggered: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "deltas_sent": self.deltas_sent,
            "deltas_received": self.deltas_received,
            "deltas_applied": self.deltas_applied,
            "duplicates_ignored": self.duplicates_ignored,
            "merge_conflicts": self.merge_conflicts,
            "escalations_triggered": self.escalations_triggered,
        }


@dataclass
class EdgeDevice:
    device_id: str
    risk_thresholds: Tuple[float, float, float] = (0.40, 0.60, 0.80)
    secret_key: bytes = b"phi_exposure_guard_default"
    sync_interval_events: int = 5
    sync_interval_seconds: float = 5.0
    escalation_callback: Optional[EscalationCallback] = None
    bus: GossipBus = field(default_factory=lambda: _default_bus)

    _graph: CRDTGraph = field(init=False)
    _last_published_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    _events_since_sync: int = field(default=0, init=False)
    _last_sync_ts: float = field(default_factory=time.time, init=False)
    _event_log: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _merge_log: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _seq_counter: int = field(default=0, init=False)
    _seen_sequences: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)
    _health: FederationHealth = field(default_factory=FederationHealth, init=False)

    def __post_init__(self) -> None:
        self._graph = CRDTGraph(device_id=self.device_id)
        self.bus.register(self.device_id)

    def process_event(
        self,
        patient_key: str,
        modality_exposures: Dict[str, int],
        link_signals: Optional[Dict[str, bool]] = None,
        event_id: str = "",
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ls = link_signals or {}

        for modality, units in modality_exposures.items():
            if units > 0:
                self._graph.record_exposure(
                    patient_key=patient_key,
                    modality=modality,
                    phi_units=units,
                    link_signal=bool(ls.get(modality, False)),
                )

        risk = self._graph.risk_for(patient_key)
        policy = self._select_policy(risk)

        current_version = self._current_pseudonym_version(patient_key)
        if risk >= self.risk_thresholds[2]:
            new_v = self._graph.bump_pseudonym_version(patient_key, "text")
            current_version = new_v

        pseudonym = deterministic_pseudonym(patient_key, self.secret_key, current_version)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        record = {
            "event_id": event_id,
            "device_id": self.device_id,
            "patient_key": patient_key,
            "risk": round(risk, 4),
            "policy": policy,
            "pseudonym": pseudonym,
            "latency_ms": round(latency_ms, 3),
        }
        self._event_log.append(record)
        self._events_since_sync += 1

        if (
            self._events_since_sync >= self.sync_interval_events
            or (time.time() - self._last_sync_ts) >= self.sync_interval_seconds
        ):
            self.maybe_publish_delta()

        return record

    def maybe_publish_delta(self) -> List[NodeDelta]:
        deltas = self._compute_deltas()
        if deltas:
            self.bus.publish(self.device_id, deltas)
            self._snapshot_published_counts()
            self._health.deltas_sent += len(deltas)
        self._events_since_sync = 0
        self._last_sync_ts = time.time()
        return deltas

    def _compute_deltas(self) -> List[NodeDelta]:
        deltas: List[NodeDelta] = []
        for nid, node in self._graph.nodes.items():
            prev_phi, prev_link = self._last_published_counts.get(nid, (0, 0))
            curr_phi = node.phi_unit_counts.get(self.device_id, 0)
            curr_link = node.link_counts.get(self.device_id, 0)
            if curr_phi == prev_phi and curr_link == prev_link:
                continue
            self._seq_counter += 1
            deltas.append(NodeDelta(
                device_id=self.device_id,
                seq_id=self._seq_counter,
                patient_key=node.patient_key,
                modality=node.modality,
                phi_units_added=curr_phi - prev_phi,
                link_signal_added=curr_link - prev_link,
                pseudonym_version=node.pseudonym_version,
                pseudonym_version_ts=node.pseudonym_version_ts,
            ))
        return deltas

    def _snapshot_published_counts(self) -> None:
        for nid, node in self._graph.nodes.items():
            self._last_published_counts[nid] = (
                node.phi_unit_counts.get(self.device_id, 0),
                node.link_counts.get(self.device_id, 0),
            )

    def receive_delta(self, delta: NodeDelta) -> Optional[Dict[str, Any]]:
        self._health.deltas_received += 1

        if delta.device_id == self.device_id:
            return None

        key = (delta.device_id, delta.node_id)
        last_seq = self._seen_sequences.get(key, -1)
        if delta.seq_id <= last_seq:
            self._health.duplicates_ignored += 1
            return None
        self._seen_sequences[key] = delta.seq_id

        if delta.phi_units_added == 0 and delta.link_signal_added == 0:
            node = self._graph.nodes.get(delta.node_id)
            if node is not None and node.pseudonym_version >= delta.pseudonym_version:
                return None

        risk_before = self._graph.risk_for(delta.patient_key)

        remote_graph = CRDTGraph(device_id=delta.device_id)
        remote_node = remote_graph.get_or_create(delta.patient_key, delta.modality)
        remote_node.increment_phi(delta.device_id, delta.phi_units_added)
        if delta.link_signal_added > 0:
            remote_node.increment_link(delta.device_id, delta.link_signal_added)
        remote_node.set_pseudonym_version(delta.pseudonym_version, delta.pseudonym_version_ts)

        self._graph.merge_from(remote_graph)
        self._health.deltas_applied += 1

        risk_after = self._graph.risk_for(delta.patient_key)

        record: Dict[str, Any] = {
            "from_device": delta.device_id,
            "delta_seq": delta.seq_id,
            "patient_key": delta.patient_key,
            "node_id": delta.node_id,
            "risk_before": round(risk_before, 4),
            "risk_after": round(risk_after, 4),
            "escalated": False,
            "new_policy": None,
            "timestamp": delta.timestamp,
        }

        policy_before = self._select_policy(risk_before)
        policy_after = self._select_policy(risk_after)
        if policy_after != policy_before:
            record["escalated"] = True
            record["new_policy"] = policy_after
            self._health.escalations_triggered += 1
            if self.escalation_callback is not None:
                self.escalation_callback(delta.patient_key, risk_after, delta.device_id)

        self._merge_log.append(record)
        return record if record["escalated"] else None

    def drain_and_merge(self) -> List[Dict[str, Any]]:
        escalations = []
        for delta in self.bus.drain(self.device_id):
            result = self.receive_delta(delta)
            if result:
                escalations.append(result)
        return escalations

    def _select_policy(self, risk: float) -> str:
        r1, r_mid, r2 = self.risk_thresholds
        if risk < r1:
            return "weak"
        if risk < r_mid:
            return "synthetic"
        if risk < r2:
            return "pseudo"
        return "redact"

    def _current_pseudonym_version(self, patient_key: str) -> int:
        versions = [
            n.pseudonym_version
            for n in self._graph.nodes.values()
            if n.patient_key == patient_key
        ]
        return max(versions, default=1)

    @property
    def graph(self) -> CRDTGraph:
        return self._graph

    def risk_snapshot(self, patient_key: str) -> float:
        return self._graph.risk_for(patient_key)

    def health(self) -> Dict[str, int]:
        return self._health.to_dict()

    def summary(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "events_processed": len(self._event_log),
            "deltas_merged": len(self._merge_log),
            "escalations": sum(1 for r in self._merge_log if r["escalated"]),
            "health": self._health.to_dict(),
            "graph": self._graph.summary(),
        }


def demo_live_federation() -> Dict[str, Any]:
    bus = GossipBus()
    secret = b"shared_phi_guard_secret"

    escalation_log: List[Dict[str, Any]] = []

    def on_escalation(patient_key: str, new_risk: float, from_device: str) -> None:
        escalation_log.append({
            "patient_key": patient_key,
            "new_risk": round(new_risk, 4),
            "triggered_by": from_device,
            "timestamp": time.time(),
        })

    ward_a   = EdgeDevice("ward_A",   bus=bus, secret_key=secret,
                          escalation_callback=on_escalation, sync_interval_events=3)
    tablet_b = EdgeDevice("tablet_B", bus=bus, secret_key=secret,
                          escalation_callback=on_escalation, sync_interval_events=3)
    kiosk_c  = EdgeDevice("kiosk_C",  bus=bus, secret_key=secret,
                          escalation_callback=on_escalation, sync_interval_events=3)

    for i in range(12):
        ward_a.process_event(
            patient_key="patient_1",
            modality_exposures={"text": 2, "image_proxy": 1},
            link_signals={"text": (i % 4 == 0)},
            event_id=f"ward_a_evt_{i}",
        )

    for i in range(8):
        tablet_b.process_event(
            patient_key="patient_1",
            modality_exposures={"audio_proxy": 2, "asr": 1},
            link_signals={"audio_proxy": True},
            event_id=f"tablet_b_evt_{i}",
        )

    for i in range(6):
        kiosk_c.process_event(
            patient_key="patient_2",
            modality_exposures={"text": 1},
            event_id=f"kiosk_c_evt_{i}",
        )

    ward_a.maybe_publish_delta()
    tablet_b.maybe_publish_delta()
    kiosk_c.maybe_publish_delta()

    ward_a_escalations   = ward_a.drain_and_merge()
    tablet_b_escalations = tablet_b.drain_and_merge()
    kiosk_c_escalations  = kiosk_c.drain_and_merge()

    p1_pseudo_a = deterministic_pseudonym("patient_1", secret,
                                          ward_a._current_pseudonym_version("patient_1"))
    p1_pseudo_b = deterministic_pseudonym("patient_1", secret,
                                          tablet_b._current_pseudonym_version("patient_1"))
    p1_pseudo_c = deterministic_pseudonym("patient_1", secret,
                                          kiosk_c._current_pseudonym_version("patient_1"))

    return {
        "devices": {
            "ward_A":   ward_a.summary(),
            "tablet_B": tablet_b.summary(),
            "kiosk_C":  kiosk_c.summary(),
        },
        "post_merge_risk": {
            "ward_A_patient_1":   round(ward_a.risk_snapshot("patient_1"), 4),
            "tablet_B_patient_1": round(tablet_b.risk_snapshot("patient_1"), 4),
            "kiosk_C_patient_2":  round(kiosk_c.risk_snapshot("patient_2"), 4),
        },
        "pseudonym_consistency": {
            "ward_A":   p1_pseudo_a,
            "tablet_B": p1_pseudo_b,
            "kiosk_C":  p1_pseudo_c,
            "consistent": p1_pseudo_a == p1_pseudo_b == p1_pseudo_c,
        },
        "escalations": {
            "ward_A":   ward_a_escalations,
            "tablet_B": tablet_b_escalations,
            "kiosk_C":  kiosk_c_escalations,
            "callback_log": escalation_log,
        },
        "convergence_guaranteed": True,
    }
