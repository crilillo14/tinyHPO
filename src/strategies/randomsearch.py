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
        
        self.iterations = iterations 

        self.param_grid = param_grid
        self.keys = list(param_grid.keys())
        self.randomvec = self.get_rand_vec(iterations)

        # Get unique seed if not provided
        if seed is None:
            seed = (os.getpid() * int(time.time()) % 2**32)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.history = []
        self.curriteration = 0
    
    def get_rand_vec(self, iterations):
        return np.random.randint(0, len(self.param_grid), size=iterations)
    
    def __call__(self, iterations : int) -> np.ndarray:
        """Return all n iterations of random hyperparameter sets."""
        
        # need to get n samples of H_n.
        
        return np.array([self.param_grid[self.randomvec[i]] for i in range(iterations)])
    
    def __iter__(self):
        return iter(self.__call__(self.iterations))
        
            