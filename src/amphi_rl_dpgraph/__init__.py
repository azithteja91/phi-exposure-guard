from __future__ import annotations

import sys
import importlib

# Import the real implementation package 
import amphi_rl_dpgraph as _impl

# Re-export the public API
from amphi_rl_dpgraph import (  # noqa: F401
    ContextState,
    RiskComponents,
    ExposurePolicyController,
    Decision,
    apply_masking,
)

__all__ = [
    "ContextState",
    "RiskComponents",
    "ExposurePolicyController",
    "Decision",
    "apply_masking",
]

# Register sub-module aliases in sys.modules

_SUBMODULES = [
    "audit_signing",
    "cmo_media",
    "cmo_registry",
    "context_state",
    "controller",
    "db",
    "dcpg",
    "dcpg_crdt",
    "eval",
    "flow_controller",
    "masking",
    "masking_ops",
    "metrics",
    "phi_detector",
    "rl_agent",
    "run_demo",
    "schemas",
]

for _submod in _SUBMODULES:
    _full_impl = f"amphi_rl_dpgraph.{_submod}"
    _full_alias = f"phi_exposure_guard.{_submod}"
    if _full_alias not in sys.modules:
        # Ensure the real module is loaded, then point the alias at it.
        _mod = importlib.import_module(_full_impl)
        sys.modules[_full_alias] = _mod
