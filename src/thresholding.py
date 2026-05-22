"""
Nonparametric dynamic thresholding for telemetry prediction errors.

This is a faithful reimplementation of the unsupervised thresholding approach
from Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding" (NASA JPL, KDD 2018). No labels are used
at threshold time -- anomalies are found purely from the structure of the
prediction-error signal, which is what makes it deployable on live telemetry.
"""

import numpy as np


def smooth_errors(errors, window=None):
    """EWMA-smooth the raw prediction errors to suppress single-step spikes."""
    errors = np.asarray(errors, dtype=float)
    if window is None:
        window = int(np.clip(0.004 * len(errors), 15, 60))
    # exponentially weighted moving average
    alpha = 2.0 / (window + 1)
    out = np.empty_like(errors)
    out[0] = errors[0]
    for i in range(1, len(errors)):
        out[i] = alpha * errors[i] + (1 - alpha) * out[i - 1]
    return out


def _prune(seqs, e_s, thresh, min_sep=0.10):
    """Keep candidate sequences whose peak smoothed-error stands clearly above
    the largest *normal* (sub-threshold) error. Removes marginal sequences that
    barely cross the threshold, without chain-deleting strong ones."""
    if not seqs:
        return seqs
    normal = e_s[e_s <= thresh]
    base = float(normal.max()) if len(normal) else float(thresh)
    cutoff = base * (1 + min_sep)
    kept = [(s, e) for (s, e) in seqs if e_s[s:e + 1].max() >= cutoff]
    # never return empty if there was a clear single strongest sequence
    if not kept:
        strongest = max(seqs, key=lambda se: e_s[se[0]:se[1] + 1].max())
        kept = [strongest]
    return kept


def find_anomalies(errors, z_range=(2.0, 18.0), z_steps=50, prune=True):
    """
    Given a 1-D array of prediction errors, return anomalous index sequences
    as a list of (start, end) tuples, fully unsupervised.

    Threshold z is chosen to maximize the proportional drop in mean & std of
    the error signal when values above the threshold are removed, normalised
    by the number of anomalies induced -- exactly the argmax objective from
    the JPL paper.
    """
    e_s = smooth_errors(errors)
    mu, sigma = np.mean(e_s), np.std(e_s)
    if sigma == 0:
        return []

    best_z, best_score = z_range[0], -np.inf
    for z in np.linspace(z_range[0], z_range[1], z_steps):
        thresh = mu + z * sigma
        above = e_s > thresh
        if not above.any():
            continue
        kept = e_s[~above]
        if len(kept) == 0:
            continue
        d_mu = (mu - kept.mean()) / mu
        d_sigma = (sigma - kept.std()) / sigma if sigma else 0
        # count contiguous anomalous regions
        n_regions = _count_regions(above)
        n_anom = above.sum()
        denom = n_anom + n_regions ** 2
        if denom == 0:
            continue
        score = (d_mu + d_sigma) / denom
        if score > best_score:
            best_score, best_z = score, z

    thresh = mu + best_z * sigma
    above = e_s > thresh
    seqs = _regions(above)
    if prune and seqs:
        seqs = _prune(seqs, e_s, thresh)
    return seqs


def _count_regions(mask):
    return len(_regions(mask))


def _regions(mask):
    """Contiguous True runs -> list of (start_idx, end_idx) inclusive."""
    regions, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(mask) - 1))
    return regions
