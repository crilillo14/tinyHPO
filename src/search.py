"""
            -- search.py --

Factory function for HPO search strategies.
"""


from typing import Dict, List, Any, Optional
from src.types import ParameterSpace 
from src.strategies.bayesian.kernels import Kernel, SquaredExponential

from src.strategies.gridsearch import GridSearch
from src.strategies.randomsearch import RandomSearch
from src.strategies.bayesiansearch import BayesianSearch



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
        return BayesianSearch
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: 'grid', 'random', 'bayesian'")