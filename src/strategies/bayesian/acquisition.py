"""
            -- acquisition.py --

Acquisition functions for Bayesian Optimization.

These functions balance exploration (high uncertainty) vs exploitation
(high predicted value) when selecting the next point to evaluate.
"""

import numpy as np
from scipy.stats import norm
from typing import Callable


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition function.

    UCB(x) = mu(x) + kappa * sigma(x)

    Args:
        mu: Predicted mean values
        sigma: Predicted standard deviations
        kappa: Exploration-exploitation tradeoff parameter (higher = more exploration)

    Returns:
        UCB values for each point
    """
    return mu + kappa * sigma


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement (EI) acquisition function.

    EI(x) = (mu(x) - y_best - xi) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (mu(x) - y_best - xi) / sigma(x)

    Args:
        mu: Predicted mean values
        sigma: Predicted standard deviations
        y_best: Best observed value so far
        xi: Exploration parameter (higher = more exploration)

    Returns:
        EI values for each point
    """
    sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero

    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0  # No improvement where we have no uncertainty

    return ei


def probability_of_improvement(mu: np.ndarray, sigma: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
    """
    Probability of Improvement (PI) acquisition function.

    PI(x) = Phi((mu(x) - y_best - xi) / sigma(x))

    Args:
        mu: Predicted mean values
        sigma: Predicted standard deviations
        y_best: Best observed value so far
        xi: Exploration parameter

    Returns:
        PI values for each point
    """
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_best - xi) / sigma
    return norm.cdf(Z)


def lower_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """
    Lower Confidence Bound (LCB) acquisition function.
    For minimization problems.

    LCB(x) = mu(x) - kappa * sigma(x)

    Args:
        mu: Predicted mean values
        sigma: Predicted standard deviations
        kappa: Exploration-exploitation tradeoff parameter

    Returns:
        LCB values for each point (lower is better for minimization)
    """
    return mu - kappa * sigma


ACQUISITION_FUNCTIONS = {
    'ucb': upper_confidence_bound,
    'ei': expected_improvement,
    'pi': probability_of_improvement,
    'lcb': lower_confidence_bound,
}


def get_acquisition_function(name: str) -> Callable:
    """Get acquisition function by name."""
    if name not in ACQUISITION_FUNCTIONS:
        raise ValueError(f"Unknown acquisition function: {name}. "
                        f"Available: {list(ACQUISITION_FUNCTIONS.keys())}")
    return ACQUISITION_FUNCTIONS[name]
