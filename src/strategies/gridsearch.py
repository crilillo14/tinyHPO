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


class GridSearch:
    """Exhaustive grid search over parameter space."""

    def __init__(self, param_grid: ParameterSpace)-> None:
        self.param_grid = param_grid
        self.keys = list(param_grid.keys())
        self.values = list(param_grid.values())

        # Calculate grid shape and total combinations
        self.grid_shape: list[float] = [len(v) for v in self.values]
        
        self.num_elements = int(np.prod(self.grid_shape))
        self.degrees_of_freedom = len(param_grid)

        # Warn about empty parameters
        for key, vals in param_grid.items():
            if len(vals) == 0:
                print(f"Warning: empty parameter array detected for '{key}'")

        # Iterator state
        self.current_indices = [0] * self.degrees_of_freedom
        self.iteration = 0
        self.history = []

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all parameter combinations as a list of dicts."""
        all_combinations = []
        for combo in product(*self.values):
            param_dict = dict(zip(self.keys, combo))
            all_combinations.append(param_dict)
        return all_combinations

    def _get_next(self) -> Dict[str, Any] | None:
        """Get next parameter combination."""
        if self.iteration >= self.num_elements:
            return None

        # Build parameter dict from current indices
        params = {}
        for i, key in enumerate(self.keys):
            params[key] = self.values[i][self.current_indices[i]]

        # Increment indices (odometer-style)
        self._increment_indices()
        self.iteration += 1

        return params

    def _increment_indices(self) -> None:
        """Increment indices like an odometer."""
        for i in range(self.degrees_of_freedom):
            self.current_indices[i] += 1
            if self.current_indices[i] < self.grid_shape[i]:
                break
            self.current_indices[i] = 0

    def __call__(self) -> Dict[str, Any] | None:
        """Return next hyperparameter set."""
        params = self._get_next()

        if params is None:
            print("Grid search completed - reached last element")
            return None

        self.history.append(params)
        return params

    def reset(self) -> None:
        """Reset the iterator to the beginning."""
        self.current_indices = [0] * self.degrees_of_freedom
        self.iteration = 0
        self.history = []

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Allow iteration over all parameter combinations."""
        self.reset()
        return self

    def __next__(self) -> Dict[str, Any]:
        params = self._get_next()
        if params is None:
            raise StopIteration
        self.history.append(params)
        return params

    def __len__(self) -> int:
        return self.num_elements

    def __repr__(self) -> str:
        return f"GridSearch(params={self.keys}, total={self.num_elements}, current={self.iteration})"
