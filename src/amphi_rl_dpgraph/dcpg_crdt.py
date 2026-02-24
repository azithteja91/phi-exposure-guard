from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CRDTNodeState:
    """
    Grow-only counter node for federated DCPG.

    Each counter field is a G-Counter: values only increase and merge takes the max.
    LWW fields use the highest timestamp.
    """
    patient_key: str
    modality: str

    # G-counters: device_id -> count
    phi_unit_counts: Dict[str, int] = field(default_factory=dict)
    link_counts: Dict[str, int] = field(default_factory=dict)

    # last-write-wins fields (higher timestamp wins)
    pseudonym_version: int = 0
    pseudonym_version_ts: float = 0.0

    risk_entropy: float = 0.0
    risk_entropy_ts: float = 0.0

    @property
    def node_id(self) -> str:
        return f"{self.patient_key}::{self.modality}"

    @property
    def total_phi_units(self) -> int:
        return sum(self.phi_unit_counts.values())

    @property
    def total_link_signals(self) -> int:
        return sum(self.link_counts.values())

    def increment_phi(self, device_id: str, units: int = 1) -> None:
        inc = max(0, int(units))
        if inc:
            self.phi_unit_counts[device_id] = self.phi_unit_counts.get(device_id, 0) + inc

    def increment_link(self, device_id: str, count: int = 1) -> None:
        inc = max(0, int(count))
        if inc:
            self.link_counts[device_id] = self.link_counts.get(device_id, 0) + inc

    def set_pseudonym_version(self, version: int, ts: Optional[float] = None) -> None:
        t = float(time.time() if ts is None else ts)
        if t >= self.pseudonym_version_ts:
            self.pseudonym_version = int(version)
            self.pseudonym_version_ts = t

    def set_risk_entropy(self, risk: float, ts: Optional[float] = None) -> None:
        t = float(time.time() if ts is None else ts)
        if t >= self.risk_entropy_ts:
            self.risk_entropy = float(risk)
            self.risk_entropy_ts = t


def merge_node(a: CRDTNodeState, b: CRDTNodeState) -> CRDTNodeState:
    """
    Merge two CRDTNodeState instances.

    G-Counter fields take element-wise max; LWW fields take highest timestamp.
    """
    if a.patient_key != b.patient_key or a.modality != b.modality:
        raise ValueError("Can only merge nodes with matching patient_key and modality")

    merged = CRDTNodeState(patient_key=a.patient_key, modality=a.modality)

    for dev in set(a.phi_unit_counts) | set(b.phi_unit_counts):
        merged.phi_unit_counts[dev] = max(a.phi_unit_counts.get(dev, 0), b.phi_unit_counts.get(dev, 0))

    for dev in set(a.link_counts) | set(b.link_counts):
        merged.link_counts[dev] = max(a.link_counts.get(dev, 0), b.link_counts.get(dev, 0))

    if a.pseudonym_version_ts >= b.pseudonym_version_ts:
        merged.pseudonym_version = a.pseudonym_version
        merged.pseudonym_version_ts = a.pseudonym_version_ts
    else:
        merged.pseudonym_version = b.pseudonym_version
        merged.pseudonym_version_ts = b.pseudonym_version_ts

    if a.risk_entropy_ts >= b.risk_entropy_ts:
        merged.risk_entropy = a.risk_entropy
        merged.risk_entropy_ts = a.risk_entropy_ts
    else:
        merged.risk_entropy = b.risk_entropy
        merged.risk_entropy_ts = b.risk_entropy_ts

    return merged


@dataclass
class CRDTGraph:
    """
    In-memory CRDT graph of patient nodes.
    Supports local updates and merge with a remote peer's graph.
    """
    device_id: str
    nodes: Dict[str, CRDTNodeState] = field(default_factory=dict)

    def get_or_create(self, patient_key: str, modality: str) -> CRDTNodeState:
        nid = f"{patient_key}::{modality}"
        node = self.nodes.get(nid)
        if node is None:
            node = CRDTNodeState(patient_key=patient_key, modality=modality)
            self.nodes[nid] = node
        return node

    def record_exposure(
        self,
        patient_key: str,
        modality: str,
        phi_units: int = 1,
        link_signal: bool = False,
    ) -> None:
        node = self.get_or_create(patient_key, modality)
        node.increment_phi(self.device_id, phi_units)
        if link_signal:
            node.increment_link(self.device_id, 1)

    def bump_pseudonym_version(self, patient_key: str, modality: str) -> int:
        node = self.get_or_create(patient_key, modality)
        new_v = node.pseudonym_version + 1
        node.set_pseudonym_version(new_v)
        return new_v

    def merge_from(self, remote: "CRDTGraph") -> int:
        updated = 0
        for nid, remote_node in remote.nodes.items():
            local = self.nodes.get(nid)
            if local is None:
                self.nodes[nid] = CRDTNodeState(
                    patient_key=remote_node.patient_key,
                    modality=remote_node.modality,
                    phi_unit_counts=dict(remote_node.phi_unit_counts),
                    link_counts=dict(remote_node.link_counts),
                    pseudonym_version=remote_node.pseudonym_version,
                    pseudonym_version_ts=remote_node.pseudonym_version_ts,
                    risk_entropy=remote_node.risk_entropy,
                    risk_entropy_ts=remote_node.risk_entropy_ts,
                )
            else:
                self.nodes[nid] = merge_node(local, remote_node)
            updated += 1
        return updated

    def risk_for(self, patient_key: str, provisional_k: float = 0.30) -> float:
        patient_nodes = [n for n in self.nodes.values() if n.patient_key == patient_key]
        if not patient_nodes:
            return 0.0

        total_units = sum(n.total_phi_units for n in patient_nodes)
        degree = len({n.modality for n in patient_nodes})

        k = float(provisional_k)
        units_factor = 1.0 - math.exp(-k * float(total_units))
        r = 1.0 - math.exp(-k * float(total_units)) * units_factor / (degree + 1.0)
        return float(max(0.0, min(1.0, r)))

    def summary(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "node_count": len(self.nodes),
            "nodes": [
                {
                    "node_id": nid,
                    "total_phi_units": n.total_phi_units,
                    "total_link_signals": n.total_link_signals,
                    "pseudonym_version": n.pseudonym_version,
                    "risk_entropy": round(n.risk_entropy, 4),
                }
                for nid, n in self.nodes.items()
            ],
        }


def demo_federated_merge() -> Dict[str, Any]:
    """
    Demonstrates two edge devices independently recording exposures,
    then merging their CRDT graphs to produce a unified risk view.
    """
    device_a = CRDTGraph(device_id="edge_device_A")
    device_b = CRDTGraph(device_id="edge_device_B")

    device_a.record_exposure("patient_1", "text", phi_units=3)
    device_a.record_exposure("patient_1", "image_proxy", phi_units=1)
    device_a.bump_pseudonym_version("patient_1", "text")

    device_b.record_exposure("patient_1", "audio_proxy", phi_units=2)
    device_b.record_exposure("patient_1", "text", phi_units=1, link_signal=True)

    merged_a = CRDTGraph(device_id="edge_device_A")
    merged_a.nodes = {
        k: CRDTNodeState(
            patient_key=v.patient_key,
            modality=v.modality,
            phi_unit_counts=dict(v.phi_unit_counts),
            link_counts=dict(v.link_counts),
            pseudonym_version=v.pseudonym_version,
            pseudonym_version_ts=v.pseudonym_version_ts,
            risk_entropy=v.risk_entropy,
            risk_entropy_ts=v.risk_entropy_ts,
        )
        for k, v in device_a.nodes.items()
    }
    nodes_updated = merged_a.merge_from(device_b)

    return {
        "device_a_before": device_a.summary(),
        "device_b": device_b.summary(),
        "merged": merged_a.summary(),
        "nodes_updated": nodes_updated,
        "merged_risk_patient_1": round(merged_a.risk_for("patient_1"), 4),
        "convergence_guaranteed": True,
    }
