#           -- gridsearch.py --

# Search a given parameter space exhaustively.
# Usage:
#   search = GridSearch(param_grid)
#   params = search()  # get next parameter set
#   all_params = search.get_all()  # get all combinations

from typing import Any, List, Dict, Iterator
import numpy as np
from itertools import product
from src.types import ParameterSpace

class GridSearch(SearchStrategy):
    def __init__(self, parameterspace, n_iterations):
        super().__init__(parameterspace, n_iterations)
        self._grid = self._build_grid()
        self._index = 0

    def suggest(self) -> Dict[str, Any]:
        config = self._grid[self._index]
        self._index += 1
        return config

    def update(self, config: Dict[str, Any], score: float) -> None:
        self._iteration += 1
        if self._best_score is None or score < self._best_score:
            self._best_config = config
            self._best_score = score

    @property
    def done(self) -> bool:
        return self._index >= len(self._grid)


