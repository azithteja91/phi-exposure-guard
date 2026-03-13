
# Rolling utility monitor for downstream task quality (e.g. masked-text classifier
# AUC) relative to a moving baseline. The first baseline_events scores establish
# the reference mean and are excluded from delta calculations, preventing early-run
# variance from registering as degradation. Call reset_baseline() after a policy
# transition to re-anchor the reference point for the new regime.

from __future__ import annotations

from collections import deque
from typing import List, Optional


class RollingUtilityMonitor:
    def __init__(self, window: int = 32, baseline_events: int = 8) -> None:
        self.window = window
        self.baseline_events = baseline_events
        self._scores: deque = deque(maxlen=window)
        self._baseline_buf: List[float] = []
        self._baseline: Optional[float] = None

    def update(self, score: float) -> None:
        if self._baseline is None:
            self._baseline_buf.append(score)
            if len(self._baseline_buf) >= self.baseline_events:
                self._baseline = sum(self._baseline_buf) / len(self._baseline_buf)
        else:
            self._scores.append(score)

    def utility_delta(self) -> float:
        if self._baseline is None or not self._scores:
            return 0.0
        return (sum(self._scores) / len(self._scores)) - self._baseline

    def mean_score(self) -> float:
        if not self._scores:
            return float(self._baseline) if self._baseline is not None else 0.0
        return sum(self._scores) / len(self._scores)

    def reset_baseline(self) -> None:
        self._baseline_buf = []
        self._baseline = None
        self._scores.clear()

    def confidence_drift(
        self,
        probs_orig: List[float],
        probs_masked: List[float],
    ) -> float:
        if not probs_orig or not probs_masked:
            return 0.0
        n = min(len(probs_orig), len(probs_masked))
        return sum(abs(a - b) for a, b in zip(probs_orig[:n], probs_masked[:n])) / n
