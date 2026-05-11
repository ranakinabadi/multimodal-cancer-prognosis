"""
Calibration plot: predicted risk vs observed event rate.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from ..evaluation.metrics import calibration_data


def plot_calibration(risk_scores: np.ndarray,
                      events:      np.ndarray,
                      n_bins:      int = 5,
                      save_path:   Optional[str | Path] = None,
                      title:       str = "Calibration plot") -> plt.Figure:
    """
    Plot normalised predicted risk vs observed event rate across bins.

    Args:
        risk_scores : (N,) predicted log-hazard ratios
        events      : (N,) event indicators
        n_bins      : number of equal-frequency bins
        save_path   : optional path to save figure
        title       : plot title

    Returns:
        matplotlib Figure
    """
    cal = calibration_data(risk_scores, events, n_bins=n_bins)

    pred_norm      = cal["pred_norm"]
    obs_event_rate = cal["obs_event_rate"]
    bin_sizes      = cal["bin_sizes"]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.scatter(
        pred_norm, obs_event_rate,
        s=[b * 40 for b in bin_sizes],
        c=range(len(bin_sizes)),
        cmap="RdYlGn_r",
        zorder=3,
        edgecolors="gray",
        linewidths=0.5,
    )
    ax.plot(pred_norm, obs_event_rate, color="gray", alpha=0.4, zorder=2)

    for x, y, n in zip(pred_norm, obs_event_rate, bin_sizes):
        ax.annotate(
            f"n={n}", (x, y),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    ax.set_xlabel("Normalised predicted risk (binned)")
    ax.set_ylabel("Observed event rate")
    ax.set_title(f"{title}\n(point size ∝ bin size)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
