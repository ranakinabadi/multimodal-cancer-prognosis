"""
Kaplan-Meier survival curves stratified by model risk score.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_kaplan_meier(risk_scores: np.ndarray,
                       times:       np.ndarray,
                       events:      np.ndarray,
                       save_path:   Optional[str | Path] = None,
                       title:       str = "Kaplan-Meier — risk stratification") -> plt.Figure:
    """
    Split patients at the median risk score into high/low groups
    and plot Kaplan-Meier survival curves with a log-rank test.

    Requires: lifelines

    Args:
        risk_scores : (N,) predicted log-hazard ratios
        times       : (N,) survival / censoring times
        events      : (N,) event indicators (1=event, 0=censored)
        save_path   : optional path to save the figure
        title       : plot title

    Returns:
        matplotlib Figure
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        raise ImportError("Install lifelines: pip install lifelines")

    median     = np.median(risk_scores)
    high_mask  = risk_scores >= median
    low_mask   = ~high_mask

    lr = logrank_test(
        times[high_mask],  times[low_mask],
        events[high_mask], events[low_mask],
    )

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))

    kmf.fit(times[high_mask], events[high_mask], label=f"High risk (n={high_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color="coral",     ci_show=True)

    kmf.fit(times[low_mask],  events[low_mask],  label=f"Low risk  (n={low_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color="steelblue", ci_show=True)

    ax.set_title(f"{title}\nLog-rank p = {lr.p_value:.4f}")
    ax.set_xlabel("Survival time (days)")
    ax.set_ylabel("Survival probability S(t)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
