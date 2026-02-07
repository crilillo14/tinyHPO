"""
Bayesian Optimization components.
"""

from typing import Any

from .acquisition import (
    expected_improvement,
    probability_of_improvement,
    lower_confidence_bound,
    get_acquisition_function,
)


# Lazy load if running gaussianprocess directly. 
# 
def __getattr__(name: str) -> Any:
    if name == 'GaussianProcess':
        from .gaussianprocess import GaussianProcess

        return GaussianProcess
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BayesianSearch',
    'GaussianProcess',
    'expected_improvement',
    'probability_of_improvement',
    'lower_confidence_bound',
    'get_acquisition_function',
]
