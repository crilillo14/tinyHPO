"""
tinyHPO - Lightweight Hyperparameter Optimization

A minimal HPO library with support for:
- Grid Search
- Random Search
- Bayesian Optimization
- Visualization tools
"""

from .search import get_hpo_strategy
from .strategies import GridSearch, RandomSearch, BayesianSearch
from . import viz

__all__ = [
    'get_hpo_strategy',
    'GridSearch',
    'RandomSearch',
    'BayesianSearch',
    'viz',
]

__version__ = '0.1.0'
