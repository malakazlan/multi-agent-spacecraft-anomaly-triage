"""
Sequence-level scoring against the real labelled anomaly windows, following the
convention in the JPL paper:

  - A labelled anomaly sequence is a TRUE POSITIVE if any predicted anomalous
    index falls inside it (the operator is alerted in time).
  - A predicted sequence overlapping no labelled sequence is a FALSE POSITIVE.
  - A labelled sequence with no overlapping prediction is a FALSE NEGATIVE.

Precision, recall and F1 are computed over sequences across all channels.
"""

import numpy as np


def _overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def score_channel(pred_seqs, true_seqs):
    tp = sum(1 for t in true_seqs if any(_overlap(p, t) for p in pred_seqs))
    fn = len(true_seqs) - tp
    fp = sum(1 for p in pred_seqs if not any(_overlap(p, t) for t in true_seqs))
    return tp, fp, fn


def aggregate(per_channel):
    TP = sum(c[0] for c in per_channel)
    FP = sum(c[1] for c in per_channel)
    FN = sum(c[2] for c in per_channel)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": TP, "FP": FP, "FN": FN,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)}


# JPL paper reported results (telemanom, SMAP+MSL combined) for reference.
JPL_BASELINE = {"precision": 0.87, "recall": 0.80, "f1": 0.71,
                "note": "Hundman et al. 2018, LSTM + nonparam thresholding, "
                        "combined SMAP+MSL. Numbers vary by run/config."}
