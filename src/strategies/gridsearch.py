#           -- gridsearch.py --

# Search a given parameter space exhaustively.
# instantiate it as follows:
# import gridsearch
# search = gridsearch(param_grid)
# next = search()
# # or
# 2darr = search.getAll()


from typing import Any, List
import numpy as np
from itertools import product  # read docs


class GridSearch:

    def __init__(self, param_grid: dict[str, List[float]) -> None:

        self.param_grid = param_grid
        self.grid_shape = np.array([len(v) for v in param_grid.values()])
        self.degrees_of_freedom = len(param_grid)

        # if there is a 0 element in the grid shape, filter it out.
        # print warning "empty parameter array detected {key} : {params}"

        # Iterator
        self.iterator = GridSearchIterator(self.grid_shape)
        self.current_indices = self.iterator.current_indices

    def getAll(self):
        # left to implement; return a 2darray of all combinations.
        pass

    def _get_next(self):
        x = self.iterator()
        if self.iterations < self.num_elements:
            hyperparams = np.array([self.param_grid[index] for index in x])
            self.iterations += 1
        else
        return

    # should return numpy array
    def __call__(self, ) -> Any:
        hp = self._get_next()

        if hp is None:
            print("Grid search completed , reached last element")
            return None

        return hp

# -------------------------------------------------------------------------------


class GridSearchIterator:

    def __init__(self, param_grid,
                 grid_shape) -> None:

        self.param_grid = param_grid
        self.current_indices = np.zeros(len(param_grid))
        self.shape = grid_shape

        self.num_elements = np.prod(self.shape, dtype=int)
        self.iteration = 0

    def __call__(self) -> None:

        # im goated
        for i, hpindex in enumerate(self.current_indices):
            if hpindex < self.shape[i]:
                self.current_indices[i] += 1
                break
            else:
                self.current_indices[i] = 0
