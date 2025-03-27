"""
            -- randomsearch.py --

Search method for hyperparameter tuning given some parameter grid.

Only locally referential: the searcher is only aware of a map of values,
and suggests a random configuration of them. 

param_grid : Parameter grid containing parameters to be optimized. 
Of the form: 

    param_grid = {
        "hp1" : [min, max, step],
        "hp2" : [min, max, step],
        ...
    }

"""

# TODO: test full compatibility with tinygrad

import numpy as np
import time
import os
from typing import Any


class RandomSearch:

    def __init__(self,
                 param_grid) -> None:

        self.param_grid = param_grid
        self.shape = self._get_shape()

        # get unique seed
        self.seed = (os.getpid() * int(time.time()) % 2**32)
        self.rng = np.random.default_rng(self.seed)
        self.history = []

    def _get_shape(self):

        shape = {}
        for key, param in self.param_grid:
            shape[key] = len(param)

        return shape

    def __call__(self) -> Any:

        next_hyperparams = {}

        for key, param in self.param_grid:

            # get the list of params, index it with a rand num mod the size of the parameter grid.
            index = self.rng.random() % self.shape[key]
            hyperparam = param[index]

            next_hyperparams[key] = hyperparam

        return next_hyperparams
