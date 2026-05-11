"""
Evaluation metrics for survival models.
"""

import numpy as np
import torch
from typing import Optional


def concordance_index(risk_scores: torch.Tensor,
                       times:       torch.Tensor,
                       events:      torch.Tensor) -> float:
    """
    Harrell's concordance index (C-index).

    Fraction of admissible pairs (i, j) where:
        - patient i had the event (δᵢ = 1)
        - patient i's event time is shorter (tᵢ < tⱼ)
        - patient i's predicted risk is higher (rᵢ > rⱼ)

    C-index = 0.5 → random; 1.0 → perfect; 0.0 → perfectly reversed.

    Args:
        risk_scores : (N,) predicted log-hazard ratios
        times       : (N,) survival / censoring times
        events      : (N,) event indicators (1=event, 0=censored)

    Returns:
        C-index as float. Returns 0.5 if no comparable pairs exist.
    """
    risk   = risk_scores.cpu().numpy()
    t      = times.cpu().numpy()
    e      = events.cpu().numpy()

    concordant = comparable = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        for j in range(len(t)):
            if t[i] < t[j]:
                comparable += 1
                if   risk[i] > risk[j]: concordant += 1.0
                elif risk[i] == risk[j]: concordant += 0.5

    return concordant / comparable if comparable > 0 else 0.5


def brier_score(predicted_survival: np.ndarray,
                times:              np.ndarray,
                events:             np.ndarray,
                eval_time:          float) -> float:
    """
    Brier score at a fixed evaluation time.

    Measures calibration: average squared difference between
    predicted survival probability S(t) and observed binary outcome.

    Args:
        predicted_survival : (N,) predicted S(eval_time) for each patient
        times              : (N,) observed survival / censoring times
        events             : (N,) event indicators
        eval_time          : time point at which to evaluate

    Returns:
        Brier score (lower = better calibrated). Range [0, 1].
    """
    n      = len(times)
    total  = 0.0
    for i in range(n):
        if times[i] <= eval_time and events[i] == 1:
            # Event occurred before eval_time → outcome = 1 (did not survive)
            total += (predicted_survival[i] - 0.0) ** 2
        elif times[i] > eval_time:
            # Survived past eval_time → outcome = 1 (did survive)
            total += (predicted_survival[i] - 1.0) ** 2
        # Censored before eval_time → excluded (inverse probability weighting
        # would be needed for a fully corrected Brier score)
    return total / n


def calibration_data(risk_scores: np.ndarray,
                      events:      np.ndarray,
                      n_bins:      int = 5) -> dict:
    """
    Bin patients by predicted risk and compute observed event rate per bin.

    Used to produce calibration plots.

    Returns:
        {
          'pred_risk_mean' : (n_bins,) mean predicted risk per bin,
          'obs_event_rate' : (n_bins,) observed event rate per bin,
          'bin_sizes'      : (n_bins,) number of patients per bin,
          'pred_norm'      : (n_bins,) normalised predicted risk in [0, 1],
        }
    """
    bins    = np.percentile(risk_scores, np.linspace(0, 100, n_bins + 1))
    bin_ids = np.digitize(risk_scores, bins[1:-1])

    pred_risk_mean, obs_event_rate, bin_sizes = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        pred_risk_mean.append(risk_scores[mask].mean())
        obs_event_rate.append(events[mask].mean())
        bin_sizes.append(int(mask.sum()))

    pred_arr  = np.array(pred_risk_mean)
    pred_norm = pred_arr - pred_arr.min()
    denom     = pred_norm.max()
    pred_norm = pred_norm / (denom + 1e-8)

    return {
        "pred_risk_mean": pred_arr,
        "obs_event_rate": np.array(obs_event_rate),
        "bin_sizes":      bin_sizes,
        "pred_norm":      pred_norm,
    }
