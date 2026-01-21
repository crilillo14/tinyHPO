"""
            -- search.py --

Factory function for HPO search strategies.
"""

from typing import Dict, List, Any, Optional
from src.types import ParameterGrid

from strategies import GridSearch, RandomSearch, BayesianSearch



def get_hpo_strategy(strategy: str, param_grid: Optional[ParameterGrid], **kwargs):
    """
    Factory function to get an HPO search strategy.
    """
    strategy = strategy.lower()

    if strategy == "grid":
        return GridSearch(param_grid)
    elif strategy == "random":
        seed = kwargs.get('seed', None)
        return RandomSearch(param_grid, seed=seed)
    elif strategy == "bayesian":
        acquisition = kwargs.get('acquisition', 'ei')
        n_initial = kwargs.get('n_initial', 5)
        seed = kwargs.get('seed', None)
        return BayesianSearch(param_grid, acquisition=acquisition,
                             n_initial=n_initial, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: 'grid', 'random', 'bayesian'")
