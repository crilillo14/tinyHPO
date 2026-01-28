"""
            -- search.py --

Factory function for HPO search strategies.
"""


from typing import Dict, List, Any, Optional
from src.types import ParameterSpace 
from src.strategies.bayesian.kernels import Kernel, SquaredExponential

from strategies import GridSearch, RandomSearch, BayesianSearch



def get_hpo_strategy(strategy: str, param_grid: ParameterSpace, X, Y, iterations=50, seed=None, acquisitionfn='ei', kernel = SquaredExponential):
    """
    Factory function to get an HPO search strategy.
    """
    strategy = strategy.lower()

    if strategy == "grid":
        return GridSearch(param_grid)
    elif strategy == "random":
        return RandomSearch(param_grid, seed=seed, iterations=iterations)
    elif strategy == "bayesian":
        return BayesianSearch(kernel, param_grid, X, Y, searchstrategy=, acquisition=acquisition,
                             n_initial=n_initial, seed=seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: 'grid', 'random', 'bayesian'")
