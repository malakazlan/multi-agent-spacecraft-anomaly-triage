"""
End-to-end run: load channels -> per-channel forecasting + dynamic thresholding
-> multi-agent triage -> sequence-level F1 vs real labels.

Usage:
    python src/run.py --backend fast            # CPU, runs anywhere
    python src/run.py --backend lstm --epochs 35  # real benchmark (needs torch)
    python src/run.py --limit 5                  # quick smoke test on 5 channels
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from data_loader import load_labels, load_channel, load_train_signal, real_data_available
from detector import prediction_errors
from thresholding import find_anomalies
from agents import triage
from evaluate import score_channel, aggregate, JPL_BASELINE

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fast", "lstm"], default="fast")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="0 = all channels")
    args = ap.parse_args()

    synthetic = not real_data_available()
    labels = load_labels()
    if args.limit:
        labels = labels.iloc[: args.limit]

    mode = "SYNTHETIC (smoke test)" if synthetic else "REAL NASA DATA"
    print(f"\n{'='*64}\n  Spacecraft Anomaly Triage  |  mode: {mode}")
    print(f"  backend: {args.backend}  |  channels: {len(labels)}\n{'='*64}\n")

    per_channel, channel_errors, t0 = [], {}, time.perf_counter()
    for _, row in labels.iterrows():
        sig, is_syn = load_channel(row.chan_id, row.num_values,
                                   row.anomaly_sequences, seed=hash(row.chan_id) % 2**31)
        train_sig = None if is_syn else load_train_signal(row.chan_id)
        err = prediction_errors(sig, backend=args.backend, epochs=args.epochs,
                                train_signal=train_sig)
        pred = find_anomalies(err)
        tp, fp, fn = score_channel(pred, row.anomaly_sequences)
        per_channel.append((tp, fp, fn))
        channel_errors[row.chan_id] = err
        print(f"  {row.chan_id:<6} {row.spacecraft:<5} "
              f"pred={len(pred):<2} true={len(row.anomaly_sequences):<2} "
              f"TP={tp} FP={fp} FN={fn}")

    metrics = aggregate(per_channel)
    events, decision = triage(channel_errors)
    elapsed = time.perf_counter() - t0

    print(f"\n{'-'*64}\n  DETECTION (sequence-level, vs real labels)")
    for k in ("precision", "recall", "f1", "TP", "FP", "FN"):
        print(f"    {k:<10}: {metrics[k]}")
    print(f"\n  JPL baseline (reference): P={JPL_BASELINE['precision']} "
          f"R={JPL_BASELINE['recall']} F1={JPL_BASELINE['f1']}")
    print(f"\n  MULTI-AGENT TRIAGE")
    print(f"    total anomaly events : {len(events)}")
    print(f"    fleet recommendation : {decision['action']} "
          f"(conf {decision['confidence']})")
    print(f"    rationale            : {decision['rationale']}")
    print(f"    decision latency     : {decision['latency_ms']} ms")
    print(f"\n  wall-clock: {elapsed:.1f}s\n{'='*64}\n")

    if synthetic:
        print("  NOTE: metrics above are on SYNTHETIC signals (real .npy not\n"
              "  present). Run ./download_data.sh then re-run for the real\n"
              "  NASA-benchmark number.\n")

    os.makedirs(RESULTS, exist_ok=True)
    out = {"mode": mode, "backend": args.backend, "metrics": metrics,
           "jpl_baseline": JPL_BASELINE, "decision": decision,
           "n_events": len(events), "wall_clock_s": round(elapsed, 1)}
    with open(os.path.join(RESULTS, "run.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
