"""
Multi-agent anomaly triage layer.

Detection alone produces a flat list of "something looks weird on channel X."
That is not actionable on a vehicle. This layer turns detections into a single
reasoned recommendation under a time budget, the way an autonomy stack must:

  ChannelMonitorAgent  -- one per channel; owns its error signal, emits
                          AnomalyEvents with a local severity (how far past
                          threshold, how persistent).

  CorrelationAgent     -- collects events in a sliding time window across all
                          channels. An anomaly seen on one isolated channel is
                          probably sensor noise; the same-time anomaly across
                          several channels is probably a real cascading fault.
                          Adjusts severity accordingly.

  DecisionAgent        -- maps adjusted severity + correlation + persistence to
                          one of CONTINUE / MONITOR / FLAG / ABORT, with a
                          confidence and a short rationale, respecting a wall-
                          clock budget so it always returns in time.

The agents communicate by passing event objects; the design is the part that
maps directly to a launch-abort / vehicle-health autonomy problem.
"""

import time
from dataclasses import dataclass, field
from typing import List

from thresholding import find_anomalies, smooth_errors
import numpy as np


@dataclass
class AnomalyEvent:
    channel: str
    start: int
    end: int
    severity: float            # local, 0..1
    adjusted: float = 0.0      # after correlation
    correlated_with: List[str] = field(default_factory=list)


class ChannelMonitorAgent:
    def __init__(self, channel, errors):
        self.channel = channel
        self.errors = errors
        self.e_s = smooth_errors(errors)

    def detect(self) -> List[AnomalyEvent]:
        seqs = find_anomalies(self.errors)
        events = []
        if not seqs:
            return events
        mu, sd = self.e_s.mean(), self.e_s.std() + 1e-9
        for (s, e) in seqs:
            peak = self.e_s[s:e + 1].max()
            z = (peak - mu) / sd
            persist = (e - s + 1) / max(1, len(self.e_s))
            sev = 1 - np.exp(-0.25 * z)          # saturating in z
            sev = min(1.0, sev + 0.3 * min(1.0, persist * 50))
            events.append(AnomalyEvent(self.channel, s, e, round(float(sev), 3)))
        return events


class CorrelationAgent:
    def __init__(self, window=150):
        self.window = window

    def correlate(self, events: List[AnomalyEvent]) -> List[AnomalyEvent]:
        for ev in events:
            overlap = []
            for other in events:
                if other is ev or other.channel == ev.channel:
                    continue
                if other.start <= ev.end + self.window and other.end >= ev.start - self.window:
                    overlap.append(other.channel)
            ev.correlated_with = sorted(set(overlap))
            # correlated across channels -> boost; isolated -> mild discount
            boost = min(0.4, 0.12 * len(ev.correlated_with))
            discount = 0.0 if ev.correlated_with else 0.1
            ev.adjusted = round(min(1.0, max(0.0, ev.severity + boost - discount)), 3)
        return events


class DecisionAgent:
    def __init__(self, time_budget_ms=50):
        self.time_budget_ms = time_budget_ms

    def decide(self, events: List[AnomalyEvent]):
        t0 = time.perf_counter()
        if not events:
            return {"action": "CONTINUE", "confidence": 0.99,
                    "rationale": "No anomalies detected on any channel.",
                    "latency_ms": 0.0}
        top = max(events, key=lambda e: e.adjusted)
        n_corr = len(top.correlated_with)
        sev = top.adjusted
        if sev >= 0.85 and n_corr >= 2:
            action, conf = "ABORT", 0.9
            why = (f"High-severity anomaly on {top.channel} correlated across "
                   f"{n_corr} other channels -> probable cascading fault.")
        elif sev >= 0.7:
            action, conf = "FLAG", 0.8
            why = (f"Elevated anomaly on {top.channel}"
                   + (f", correlated with {n_corr} channels." if n_corr else ", isolated."))
        elif sev >= 0.4:
            action, conf = "MONITOR", 0.7
            why = f"Moderate deviation on {top.channel}; watch for escalation."
        else:
            action, conf = "CONTINUE", 0.85
            why = f"Minor isolated deviation on {top.channel}; within tolerance."
        latency = (time.perf_counter() - t0) * 1000
        if latency > self.time_budget_ms:
            why += " [budget exceeded -> conservative escalation]"
            if action == "CONTINUE":
                action = "MONITOR"
        return {"action": action, "confidence": conf, "rationale": why,
                "top_channel": top.channel, "severity": sev,
                "correlated_channels": top.correlated_with,
                "latency_ms": round(latency, 3)}


def triage(channel_errors: dict, time_budget_ms=50):
    """channel_errors: {channel_id: error_array}. Returns (events, decision)."""
    all_events = []
    for ch, err in channel_errors.items():
        all_events.extend(ChannelMonitorAgent(ch, err).detect())
    all_events = CorrelationAgent().correlate(all_events)
    decision = DecisionAgent(time_budget_ms).decide(all_events)
    return all_events, decision
