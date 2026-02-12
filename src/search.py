"""
            -- search.py --

Factory function for HPO search strategies.
"""


from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from src.types import ParameterSpace 
from src.strategies.bayesian.kernels import Kernel, RBF

from src.strategies.gridsearch import GridSearch
from src.strategies.randomsearch import RandomSearch
from src.strategies.bayesiansearch import BayesianSearch


def get_hpo_strategy(strategy: str, param_grid: ParameterSpace, X, Y, iterations=50, seed=None, acquisitionfn='ei', kernel=RBF):
    """
    Factory function to get an HPO search strategy.
    """
    strategy = strategy.lower()

    if strategy == "grid": 
        return GridSearch(param_grid, iterations)
    elif strategy == "random":
        return RandomSearch(param_grid, iterations)
    elif strategy == "bayesian":
        return BayesianSearch(param_grid, iterations, acquisitionfn, )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Available: 'grid', 'random', 'bayesian'")


class SearchStrategy(ABC):
    """
    Common interface for hyperparameter search strategies.

    The caller runs a loop:
        strategy = SomeStrategy(space, n_iterations=50, ...)
        while not strategy.done:
            config = strategy.suggest()
            score  = objective(config)
            strategy.update(config, score)
        best = strategy.best
    """

    def __init__(self, parameterspace: Dict[str, Any], n_iterations: int):
        self.parameterspace = parameterspace
        self.n_iterations = n_iterations
        self._iteration = 0
        self._best_config: Optional[Dict[str, Any]] = None
        self._best_score: Optional[float] = None

    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """Propose the next configuration to evaluate."""
        ...

    @abstractmethod
    def update(self, config: Dict[str, Any], score: float) -> None:
        """Feed back an observation (config, score) from the objective."""
        ...

    @property
    def done(self) -> bool:
        """Whether the budget is exhausted."""
        return self._iteration >= self.n_iterations

    @property
    def best(self) -> Dict[str, Any]:
        """Best configuration found so far."""
        return self._best_config or {}