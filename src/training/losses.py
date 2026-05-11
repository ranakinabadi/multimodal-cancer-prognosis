"""
Survival analysis loss functions.
"""

import torch


def cox_partial_likelihood(risk_scores: torch.Tensor,
                            times:       torch.Tensor,
                            events:      torch.Tensor) -> torch.Tensor:
    """
    Negative partial log-likelihood for the Cox proportional hazards model.

    For each patient i with an observed event (δᵢ = 1):
        ℓᵢ = rᵢ − log Σⱼ:tⱼ≥tᵢ exp(rⱼ)

    Loss = −mean(ℓᵢ) over all event patients.

    Args:
        risk_scores : (B,) predicted log-hazard ratios
        times       : (B,) survival or censoring times
        events      : (B,) event indicators (1 = event, 0 = censored)

    Returns:
        Scalar loss tensor. Returns 0 if no uncensored patients in batch.
    """
    if events.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)

    # Sort by descending survival time so the risk set is a prefix
    order       = torch.argsort(times, descending=True)
    risk_scores = risk_scores[order]
    events      = events[order]

    # log-sum-exp over risk set at each time point (numerically stable)
    log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
    loss       = -torch.mean((risk_scores - log_cumsum) * events)
    return loss
