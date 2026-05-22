"""
Loads NASA SMAP/MSL telemetry channels.

Real mode:   reads data/train/<chan>.npy and data/test/<chan>.npy
             (download with ./download_data.sh -- ~330 MB from NASA JPL's
             public bucket). Feature 0 is the telemetry value; remaining
             columns are one-hot command inputs, per the original dataset.

Synthetic fallback: if the real .npy files are absent, we synthesise a test
             signal for each channel whose anomalous regions are placed at the
             *real* labelled windows from labeled_anomalies.csv. This exists
             ONLY so the full pipeline runs end-to-end for a smoke test. Any
             metric computed on synthetic data is marked SYNTHETIC and must not
             be reported as a NASA-benchmark result. The headline number comes
             from real mode.
"""

import ast
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def load_labels():
    df = pd.read_csv(os.path.join(DATA, "labeled_anomalies.csv"))
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    return df


def real_data_available():
    return os.path.isdir(os.path.join(DATA, "test")) and any(
        f.endswith(".npy") for f in os.listdir(os.path.join(DATA, "test"))
    )


def load_channel(chan_id, n_values, anomaly_seqs, seed=0):
    """Return (test_signal_2d, is_synthetic)."""
    test_path = os.path.join(DATA, "test", f"{chan_id}.npy")
    if os.path.exists(test_path):
        return np.load(test_path), False
    return _synthesise(n_values, anomaly_seqs, seed), True


def _synthesise(n, anomaly_seqs, seed):
    """Structurally faithful synthetic telemetry: smooth baseline + periodicity
    + noise, with injected deviations exactly at the real labelled windows."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = (
        0.6 * np.sin(2 * np.pi * t / 200)
        + 0.3 * np.sin(2 * np.pi * t / 53)
        + 0.05 * rng.standard_normal(n)
    )
    # slow drift
    base += np.cumsum(rng.standard_normal(n)) * 0.002
    base = (base - base.min()) / (np.ptp(base) + 1e-9)
    for (s, e) in anomaly_seqs:
        s, e = int(s), min(int(e), n - 1)
        kind = rng.integers(0, 3)
        if kind == 0:        # level shift
            base[s:e + 1] += rng.uniform(0.4, 0.8)
        elif kind == 1:      # variance burst
            base[s:e + 1] += rng.uniform(0.3, 0.6) * rng.standard_normal(e - s + 1)
        else:                # flatline / stuck sensor
            base[s:e + 1] = base[s]
    return base.reshape(-1, 1)
