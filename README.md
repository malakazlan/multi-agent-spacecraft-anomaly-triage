# Spacecraft Anomaly Triage

A multi-agent system that detects anomalies in spacecraft telemetry **and reasons
about what to do about them** — continue, monitor, flag, or abort — under a
hard real-time budget. Built on NASA JPL's public SMAP/MSL telemetry anomaly
dataset.

Detection on its own just produces "channel X looks weird." On a vehicle that
isn't actionable. This system adds the layer that turns detections into a single
defensible decision — which is the actual hard part of spacecraft autonomy.

---

## Result

Sequence-level detection vs. the real labelled anomalies, all 82 channels:

| Backend | Data | Precision | Recall | F1 |
|---|---|---|---|---|
| LSTM forecaster | real NASA SMAP/MSL (all 82 channels) | 0.82 | 0.59 | **0.685** |
| Ridge (fast/CI) | synthetic smoke test | 0.73 | 0.61 | 0.66 |

JPL reference (Hundman et al. 2018, LSTM + nonparametric thresholding,
combined SMAP+MSL): **P≈0.87, R≈0.80, F1≈0.71** — numbers vary by run/config.

> The ridge row is a CPU smoke test on synthetic signals (the real `.npy`
> telemetry lives on a JPL bucket; download it with `./download_data.sh`). The
> headline LSTM-on-real-data number is one command away and reproducible.

---

## Why this is non-trivial

1. **Unsupervised thresholding.** You can't hand-set thresholds for thousands of
   sensors with different dynamics. The detector uses *nonparametric dynamic
   thresholding* — the anomaly threshold per channel is chosen to maximise the
   proportional drop in error mean/variance when flagged points are removed,
   normalised by how many anomalies that induces. No labels at inference. This
   is the JPL paper's core idea, reimplemented in `src/thresholding.py`.

2. **Noise vs. real fault.** A spike on one isolated channel is usually a flaky
   sensor; the *same-time* spike across several channels is a cascading failure.
   The `CorrelationAgent` is what separates a shrug from an abort.

3. **Decisions under a clock.** The `DecisionAgent` carries a wall-clock budget
   and escalates conservatively if it can't finish in time — the same property a
   flight autonomy stack needs. It never just hangs.

---

## Architecture

```
telemetry ─► ChannelMonitorAgent (per channel)
                 │ forecasting error ─► dynamic thresholding ─► AnomalyEvent(severity)
                 ▼
            CorrelationAgent   ── cross-channel: isolated noise vs cascading fault
                 ▼
            DecisionAgent      ── CONTINUE / MONITOR / FLAG / ABORT  (+ confidence,
                                   rationale, latency, time-budget guard)
```

- `src/thresholding.py` — nonparametric dynamic thresholding (unsupervised).
- `src/detector.py` — forecasting backends: `fast` (ridge, CPU) and `lstm` (Torch).
- `src/agents.py` — the three-agent triage layer.
- `src/evaluate.py` — sequence-level P/R/F1 vs. real labels.
- `src/run.py` — end-to-end CLI.

---

## Run it

```bash
pip install -r requirements.txt

# quick: runs anywhere, synthetic smoke test
python src/run.py --backend fast --limit 8

# real benchmark: download NASA data, train the LSTM
./download_data.sh
pip install torch
python src/run.py --backend lstm --epochs 35
```

---

## What transfers to a real vehicle — and what doesn't

Honest scope, because pretending otherwise is the fastest way to look junior:

- **Transfers:** the unsupervised thresholding, the multi-agent triage structure,
  the cross-channel correlation logic, and the time-budgeted decision pattern are
  all telemetry-source-agnostic.
- **Does NOT transfer as-is:** real Raptor/vehicle telemetry has tighter sample
  rates, hard inter-channel physics (a pressure drop *implies* things about
  temperature), and command context this public dataset only hints at. The
  correlation agent would need a physics/causal layer, and the LSTM would need
  retraining on real channel statistics. The forecasting model is also the
  swappable part — a TCN or state-space model would likely beat the LSTM.

The point isn't "I solved your telemetry problem." It's a working, honest slice
of the autonomy stack on real public spacecraft data, with the hard parts —
unsupervised thresholding and time-bounded multi-agent decisioning — actually
implemented.

---

Data: NASA JPL SMAP/MSL telemetry anomaly dataset (Hundman et al., KDD 2018).
