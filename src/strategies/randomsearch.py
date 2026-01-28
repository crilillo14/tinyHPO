"""
            -- randomsearch.py --

Search method for hyperparameter tuning given some parameter grid.

Only locally referential: the searcher is only aware of a map of values,
and suggests a random configuration of them.

param_grid : Parameter grid containing parameters to be optimized.
Of the form:

    param_grid = {
        "hp1" : [val1, val2, val3, ...],
        "hp2" : [val1, val2, val3, ...],
        ...
    }

"""

import numpy as np
import time
import os
from typing import Any, Dict, List, Optional
from src.types import ParameterSpace, ScalarHyperparameter, CategoricalHyperparameter

class RandomSearch:
    """Random search over parameter space."""

    def __init__(self,
                 param_grid: ParameterSpace,
                 seed: Optional[int] = None,
                 iterations : int = 50) -> None:

        self.param_grid = param_grid
        self.keys = list(param_grid.keys())
        self.randomvec = self.get_rand_vec()

        # Get unique seed if not provided
        if seed is None:
            seed = (os.getpid() * int(time.time()) % 2**32)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.history = []
        self.iteration = 0
    
    def get_rand_vec(self):
        for 
        

    def __call__(self) -> Dict[str, Any]:
        """Return a random hyperparameter set."""
        next_hyperparams = {}
         

        for key, param in self.param_grid.items():
            # Randomly select an index and get the parameter value
            index = self.rng.integers(0, self.shape[key])
            next_hyperparams[key] = param[index]

        self.history.append(next_hyperparams)
        self.iteration += 1
        return next_hyperparams

    def sample(self, n: int) -> List[Dict[str, Any]]:
        """Sample n random hyperparameter sets."""
        return [self() for _ in range(n)]

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the search with optional new seed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.history = []
        self.iteration = 0

    def __len__(self) -> int:
        """Return total number of possible combinations."""
        return int(np.prod(list(self.shape.values())))

    def __repr__(self) -> str:
        return f"RandomSearch(params={self.keys}, seed={self.seed}, sampled={self.iteration})"
