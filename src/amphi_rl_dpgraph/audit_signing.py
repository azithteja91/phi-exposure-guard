from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def generate_signing_key():
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization

        private_key = ec.generate_private_key(ec.SECP256R1())
        pub_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")
        return private_key, pub_pem
    except ImportError:
        return None, None


def sign_record(record: Dict[str, Any], private_key: Any) -> str:
    canonical = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
    if private_key is None:
        # graceful degradation on edge devices without cryptography lib
        return hashlib.sha256(canonical).hexdigest()
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes

        sig = private_key.sign(canonical, ec.ECDSA(hashes.SHA256()))
        return sig.hex()
    except Exception:
        return hashlib.sha256(canonical).hexdigest()


def verify_record(record: Dict[str, Any], signature_hex: str, public_key: Any) -> bool:
    if public_key is None:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes

        canonical = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        sig_bytes = bytes.fromhex(signature_hex)
        public_key.verify(sig_bytes, canonical, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_merkle_root(leaf_hashes: List[str]) -> str:
    if not leaf_hashes:
        return _sha256("")
    nodes = list(leaf_hashes)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])
        nodes = [_sha256(nodes[i] + nodes[i + 1]) for i in range(0, len(nodes), 2)]
    return nodes[0]


@dataclass
class MerkleCheckpoint:
    checkpoint_id: str
    timestamp: float
    record_count: int
    merkle_root: str
    leaf_hashes: List[str] = field(default_factory=list)


@dataclass
class SignedAuditEntry:
    record: Dict[str, Any]
    signature: str
    record_hash: str
    timestamp: float = field(default_factory=time.time)


class AuditChain:
    def __init__(self, private_key: Any = None, checkpoint_interval: int = 50) -> None:
        self._key = private_key
        self._checkpoint_interval = int(checkpoint_interval)
        self._entries: List[SignedAuditEntry] = []
        self._checkpoints: List[MerkleCheckpoint] = []
        self._pending_hashes: List[str] = []
        # snapshot registry for forensic replay: snapshot_id -> snapshot dict
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        # versioned CMO registry snapshots for deterministic replay
        self._cmo_registry_versions: List[Dict[str, Any]] = []

    def append(self, record: Dict[str, Any]) -> SignedAuditEntry:
        canonical = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        record_hash = hashlib.sha256(canonical).hexdigest()
        signature = sign_record(record, self._key)

        entry = SignedAuditEntry(
            record=record,
            signature=signature,
            record_hash=record_hash,
            timestamp=float(record.get("timestamp", time.time())),
        )
        self._entries.append(entry)
        self._pending_hashes.append(record_hash)

        if len(self._pending_hashes) >= self._checkpoint_interval:
            self.checkpoint()
        return entry

    def checkpoint(self) -> Optional[MerkleCheckpoint]:
        if not self._pending_hashes:
            return None
        root = build_merkle_root(self._pending_hashes)
        cp = MerkleCheckpoint(
            checkpoint_id=f"cp_{len(self._checkpoints):04d}",
            timestamp=time.time(),
            record_count=len(self._pending_hashes),
            merkle_root=root,
            leaf_hashes=list(self._pending_hashes),
        )
        self._checkpoints.append(cp)
        self._pending_hashes.clear()
        return cp

    def register_snapshot(self, snapshot_id: str, snapshot_dict: Dict[str, Any]) -> None:
        """Store a DCPG snapshot for later forensic replay lookup."""
        self._snapshots[snapshot_id] = snapshot_dict

    def register_cmo_version(self, cmo_set: List[str], policy_version: str) -> None:
        """Record which CMOs were active at a given policy version."""
        self._cmo_registry_versions.append(
            {
                "policy_version": policy_version,
                "timestamp": time.time(),
                "cmo_set": list(cmo_set),
            }
        )

    def replay(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Forensic replay: return the full audit entry for a given event_id
        alongside the DCPG snapshot and CMO registry that were active at that time.
        """
        entry = next((e for e in self._entries if e.record.get("event_id") == event_id), None)
        if entry is None:
            return None

        ts = float(entry.timestamp)

        best_snap = None
        best_snap_ts = 0.0
        for _, snap in self._snapshots.items():
            snap_ts = float(snap.get("timestamp", 0.0))
            if snap_ts <= ts and snap_ts > best_snap_ts:
                best_snap = snap
                best_snap_ts = snap_ts

        best_cmo = None
        best_cmo_ts = 0.0
        for cv in self._cmo_registry_versions:
            cv_ts = float(cv.get("timestamp", 0.0))
            if cv_ts <= ts and cv_ts > best_cmo_ts:
                best_cmo = cv
                best_cmo_ts = cv_ts

        return {
            "event_id": event_id,
            "record": entry.record,
            "signature": entry.signature,
            "record_hash": entry.record_hash,
            "dcpg_snapshot": best_snap,
            "cmo_registry_at_event": best_cmo,
        }

    def to_fhir_audit_event(self, entry: SignedAuditEntry) -> Dict[str, Any]:
        """Serialise a single audit entry as a FHIR AuditEvent resource."""
        rec = entry.record
        return {
            "resourceType": "AuditEvent",
            "id": uuid.uuid4().hex,
            "type": {
                "system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
                "code": "110110",
                "display": "Patient Record",
            },
            "subtype": [
                {
                    "system": "http://hl7.org/fhir/restful-interaction",
                    "code": "update",
                    "display": "PHI Masking",
                }
            ],
            "action": "U",
            "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry.timestamp)),
            "outcome": "0",
            "agent": [
                {
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ParticipationType",
                                "code": "AUT",
                            }
                        ]
                    },
                    "name": "adaptive_deid_system",
                    "requestor": True,
                }
            ],
            "source": {
                "observer": {"display": "AdaptiveDeidPipeline"},
                "type": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/security-source-type",
                        "code": "4",
                        "display": "Application Server",
                    }
                ],
            },
            "entity": [
                {
                    "what": {"identifier": {"value": str(rec.get("event_id", ""))}},
                    "type": {
                        "system": "http://terminology.hl7.org/CodeSystem/audit-entity-type",
                        "code": "2",
                        "display": "System Object",
                    },
                    "detail": [
                        {"type": "policy_run", "valueString": str(rec.get("policy_run", ""))},
                        {"type": "chosen_policy", "valueString": str(rec.get("chosen_policy", ""))},
                        {"type": "modality", "valueString": str(rec.get("modality", ""))},
                        {"type": "risk_score", "valueString": str(rec.get("risk", ""))},
                        {"type": "signature", "valueString": entry.signature[:32]},
                    ],
                }
            ],
        }

    def export_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self._entries:
                row = {
                    **e.record,
                    "_signature": e.signature,
                    "_record_hash": e.record_hash,
                    "_chain_ts": e.timestamp,
                }
                f.write(json.dumps(row, default=str) + "\n")

    def export_checkpoints_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for cp in self._checkpoints:
                f.write(
                    json.dumps(
                        {
                            "checkpoint_id": cp.checkpoint_id,
                            "timestamp": cp.timestamp,
                            "record_count": cp.record_count,
                            "merkle_root": cp.merkle_root,
                        },
                        default=str,
                    )
                    + "\n"
                )

    def export_fhir_jsonl(self, path: str) -> None:
        """Export all entries as FHIR AuditEvent JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for e in self._entries:
                f.write(json.dumps(self.to_fhir_audit_event(e), default=str) + "\n")

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)


@dataclass
class DCPGSnapshot:
    snapshot_id: str
    timestamp: float
    patient_keys: List[str]
    node_summaries: Dict[str, Any]
    policy_version: str
    trigger: str


def take_dcpg_snapshot(
    context_state: Any,
    patient_keys: List[str],
    policy_version: str = "v1",
    trigger: str = "interval",
    snapshot_id: Optional[str] = None,
) -> DCPGSnapshot:
    sid = snapshot_id or f"snap_{int(time.time() * 1000)}"
    node_summaries: Dict[str, Any] = {}

    for pk in patient_keys:
        try:
            comps = context_state.risk_components(pk)
            pv = context_state.get_pseudonym_version(pk)
            link_mods = context_state.link_modalities_recent(pk)
            node_summaries[pk] = {
                "pseudonym_version": pv,
                "risk": round(comps.risk, 4),
                "effective_units": comps.effective_units,
                "link_modalities": link_mods,
            }
        except Exception as exc:
            node_summaries[pk] = {"error": str(exc)}

    return DCPGSnapshot(
        snapshot_id=sid,
        timestamp=time.time(),
        patient_keys=list(patient_keys),
        node_summaries=node_summaries,
        policy_version=str(policy_version),
        trigger=str(trigger),
    )
