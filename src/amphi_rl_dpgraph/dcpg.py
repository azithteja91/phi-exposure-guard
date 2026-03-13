# DCPGAdapter bridges a live SQLite context state to graph nodes and edges
# representing per-patient PHI exposure. Nodes are built from modality-grouped
# exposure rows; edges capture co-occurrence and cross-modal link signals weighted
# by temporal decay, semantic similarity, modality distance, and trust. Embeddings
# are computed per-modality (text/ASR, audio, image) with an in-process LRU cache.

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DCPGNode:
    patient_key: str
    phi_type: str
    modality: str
    embedding: List[float] = field(default_factory=list)
    pseudonym_version: int = 0
    risk_entropy: float = 0.0
    context_confidence: float = 1.0

    @property
    def node_id(self) -> str:
        return f"{self.patient_key}::{self.modality}::{self.phi_type}"


@dataclass
class DCPGEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 0.0

    @classmethod
    def compute_weight(
        cls,
        f_temporal: float,
        f_semantic: float,
        f_modality: float,
        f_trust: float,
        alpha: float = 0.30,
        beta: float = 0.30,
        gamma: float = 0.25,
        delta: float = 0.15,
    ) -> float:
        return (
            alpha * f_temporal
            + beta * f_semantic
            + gamma * f_modality
            + delta * f_trust
        )


def _ngram_vector(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    s = str(text).lower()
    for i in range(len(s) - 1):
        h = (ord(s[i]) * 31 + ord(s[i + 1])) % dim
        vec[h] += 1.0
    total = sum(vec) or 1.0
    return [v / total for v in vec]


def _text_embedding(text: str) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer

        model = _text_embedding._cache.get("st")
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _text_embedding._cache["st"] = model
        return model.encode(str(text), show_progress_bar=False).tolist()
    except Exception:
        return _ngram_vector(text, dim=64)


_text_embedding._cache: Dict[str, Any] = {}


def _mfcc_embedding(payload: Any) -> List[float]:
    try:
        import numpy as np

        arr = np.asarray(payload, dtype=np.float32).flatten()
        if arr.size == 0:
            return [0.0] * 13
        norm = float(np.linalg.norm(arr)) or 1.0
        vec = (arr[:13] / norm).tolist()
        if len(vec) < 13:
            vec.extend([0.0] * (13 - len(vec)))
        return vec
    except Exception:
        return [0.0] * 13


def _image_embedding(payload: Any) -> List[float]:
    try:
        import numpy as np

        arr = np.asarray(payload, dtype=np.float32)
        if arr.ndim < 2:
            return [float(payload)] + [0.0] * 31
        flat = arr.flatten()
        chunk = max(1, len(flat) // 32)
        vec = [float(flat[i * chunk : (i + 1) * chunk].mean()) for i in range(32)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
    except Exception:
        return [0.0] * 32


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n])) or 1.0
    nb = math.sqrt(sum(x * x for x in b[:n])) or 1.0
    return float(dot / (na * nb))


def _modality_embedding(modality: str, payload: Any) -> List[float]:
    if modality in ("text", "asr"):
        return _text_embedding(str(payload)) if payload else _ngram_vector("", 64)
    if modality in ("audio_proxy", "audio_link", "waveform_proxy"):
        return _mfcc_embedding(payload)
    if modality in ("image_proxy", "image_link"):
        return _image_embedding(payload)
    return _ngram_vector(str(payload), 32)


class DCPGAdapter:
    CROSS_MODAL_SIM_THRESHOLD = 0.30

    def __init__(self, context_state: Any) -> None:
        self.ctx = context_state
        self._emb_cache: Dict[Tuple[str, str], List[float]] = {}

    def _get_embedding(
        self,
        patient_key: str,
        modality: str,
        payload: Any = None,
    ) -> List[float]:
        key = (patient_key, modality)
        if key not in self._emb_cache:
            self._emb_cache[key] = _modality_embedding(
                modality, payload if payload is not None else ""
            )
        return self._emb_cache[key]

    def invalidate_embedding(self, patient_key: str, modality: str) -> None:
        self._emb_cache.pop((patient_key, modality), None)

    def get_nodes(self, patient_key: str) -> List[DCPGNode]:
        pk = str(patient_key)
        rows = self.ctx._conn.execute(
            """
            SELECT modality, SUM(phi_units) AS total_units
            FROM exposures
            WHERE patient_key=?
            GROUP BY modality
            """,
            (pk,),
        ).fetchall()

        comps = self.ctx.risk_components(pk)
        pv = self.ctx.get_pseudonym_version(pk)

        nodes: List[DCPGNode] = []
        for r in rows:
            mod = str(r["modality"])
            units = int(r["total_units"] or 0)
            phi_type = _modality_to_phi_type(mod)
            confidence = min(1.0, units / max(1, comps.effective_units))
            emb = self._get_embedding(pk, mod)
            nodes.append(
                DCPGNode(
                    patient_key=pk,
                    phi_type=phi_type,
                    modality=mod,
                    embedding=emb,
                    pseudonym_version=pv,
                    risk_entropy=float(comps.risk),
                    context_confidence=float(confidence),
                )
            )
        return nodes

    def get_edges(self, patient_key: str) -> List[DCPGEdge]:
        pk = str(patient_key)
        edges: List[DCPGEdge] = []

        co_rows = self.ctx._conn.execute(
            """
            SELECT e1.modality AS src_mod,
                   e2.modality AS tgt_mod,
                   e1.ts       AS ts
            FROM exposures e1
            JOIN exposures e2
              ON e1.event_id = e2.event_id
             AND e1.patient_key = e2.patient_key
             AND e1.modality < e2.modality
            WHERE e1.patient_key=?
            ORDER BY e1.ts DESC
            LIMIT 200
            """,
            (pk,),
        ).fetchall()

        seen_pairs: set[Tuple[str, str]] = set()
        now_ts = time.time()

        for r in co_rows:
            src_mod = str(r["src_mod"])
            tgt_mod = str(r["tgt_mod"])
            pair = (src_mod, tgt_mod)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            age_s = max(0.0, now_ts - float(r["ts"] or now_ts))
            half_life = float(self.ctx.recency_half_life_s)
            f_temporal = math.pow(0.5, age_s / max(1.0, half_life))

            emb_src = self._get_embedding(pk, src_mod)
            emb_tgt = self._get_embedding(pk, tgt_mod)
            sim = _cosine_similarity(emb_src, emb_tgt)
            f_semantic = float(sim) if sim > self.CROSS_MODAL_SIM_THRESHOLD else 0.5

            f_modality = 0.8 if _is_cross_modal(src_mod, tgt_mod) else 0.5
            f_trust = 0.9

            w = DCPGEdge.compute_weight(f_temporal, f_semantic, f_modality, f_trust)
            src_id = f"{pk}::{src_mod}::{_modality_to_phi_type(src_mod)}"
            tgt_id = f"{pk}::{tgt_mod}::{_modality_to_phi_type(tgt_mod)}"

            edges.append(
                DCPGEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    edge_type="co_occurrence",
                    weight=round(w, 4),
                )
            )

        link_rows = self.ctx._conn.execute(
            """
            SELECT modality, ts
            FROM exposures
            WHERE patient_key=? AND modality LIKE '%_link' AND phi_units > 0
            ORDER BY ts DESC
            LIMIT 50
            """,
            (pk,),
        ).fetchall()

        for r in link_rows:
            link_mod = str(r["modality"])
            base_mod = link_mod.replace("_link", "")
            age_s = max(0.0, now_ts - float(r["ts"] or now_ts))
            half_life = float(self.ctx.recency_half_life_s)
            f_temporal = math.pow(0.5, age_s / max(1.0, half_life))
            w = DCPGEdge.compute_weight(f_temporal, 0.7, 1.0, 0.9)
            src_id = f"{pk}::{base_mod}::{_modality_to_phi_type(base_mod)}"
            tgt_id = f"{pk}::cross_modal::LINK"
            edges.append(
                DCPGEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    edge_type="cross_modal",
                    weight=round(w, 4),
                )
            )

        return edges

    def cross_modal_match(
        self,
        patient_key: str,
        new_modality: str,
        new_payload: Any,
        threshold: float = CROSS_MODAL_SIM_THRESHOLD,
    ) -> List[str]:
        pk = str(patient_key)
        new_emb = _modality_embedding(new_modality, new_payload)
        matched: List[str] = []

        rows = self.ctx._conn.execute(
            "SELECT DISTINCT modality FROM exposures WHERE patient_key=?",
            (pk,),
        ).fetchall()

        for r in rows:
            mod = str(r["modality"])
            if mod == new_modality:
                continue
            existing_emb = self._get_embedding(pk, mod)
            sim = _cosine_similarity(new_emb, existing_emb)
            if sim >= float(threshold):
                matched.append(mod)

        return matched

    def provisional_risk(
        self,
        patient_key: str,
        now_ts: Optional[float] = None,
    ) -> float:
        pk = str(patient_key)
        comps = self.ctx.risk_components(pk, now_ts=now_ts)
        exposures = float(comps.effective_units)

        row = self.ctx._conn.execute(
            "SELECT COUNT(DISTINCT modality) AS deg FROM exposures WHERE patient_key=?",
            (pk,),
        ).fetchone()
        degree = int(row["deg"]) if row else 0

        confidence = float(comps.units_factor)
        r = 1.0 - math.exp(-0.3 * exposures) * confidence / (degree + 1)
        return float(max(0.0, min(1.0, r)))

    def graph_summary(self, patient_key: str) -> Dict[str, Any]:
        pk = str(patient_key)
        nodes = self.get_nodes(pk)
        edges = self.get_edges(pk)
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "modality": n.modality,
                    "phi_type": n.phi_type,
                    "risk_entropy": round(n.risk_entropy, 4),
                    "pseudonym_version": n.pseudonym_version,
                    "context_confidence": round(n.context_confidence, 4),
                    "embedding_dim": len(n.embedding),
                }
                for n in nodes
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type,
                    "weight": e.weight,
                }
                for e in edges
            ],
            "provisional_risk": round(self.provisional_risk(pk), 4),
        }


def _modality_to_phi_type(modality: str) -> str:
    mapping = {
        "text": "NAME_DATE_MRN_FACILITY",
        "asr": "NAME_DATE_MRN",
        "image_proxy": "FACE_IMAGE",
        "waveform_proxy": "WAVEFORM_HEADER",
        "audio_proxy": "VOICE",
        "image_link": "FACE_LINK",
        "audio_link": "VOICE_LINK",
    }
    return mapping.get(modality, modality.upper())


def _is_cross_modal(mod_a: str, mod_b: str) -> bool:
    groups = {
        "text": "text",
        "asr": "text",
        "image_proxy": "visual",
        "waveform_proxy": "signal",
        "audio_proxy": "audio",
    }
    return groups.get(mod_a, "other") != groups.get(mod_b, "other")


def _phi_type_match(mod_a: str, mod_b: str) -> bool:
    return _modality_to_phi_type(mod_a) == _modality_to_phi_type(mod_b)
