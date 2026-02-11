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
from src.search import SearchStrategy



class RandomSearch(SearchStrategy):
    def __init__(self, parameterspace, n_iterations):
        super().__init__(parameterspace, n_iterations)
    
    
    def suggest(self) -> Dict[str, Any]:
        return {name: h.sample() for name, h in self.parameterspace.items()}

    def update(self, config: Dict[str, Any], score: float) -> None:
        self._iteration += 1
        if self._best_score is None or score < self._best_score:
            self._best_config = config
            self._best_score = score