from typing import ParamSpec
import numpy as np 


class GridSearch:

    def __init__(self, param_grid) -> None:

        self.param_grid = param_grid
        self.grid_shape = np.array([len(v) for v in param_grid.values()])
        self.degrees_of_freedom = len(param_grid)

        # if there are n hparams to optimize,
        # the 0th will be the outermost loop 
        # the nth will be the innermost loop.
        
        self.iterator = GridSearchIterator(self.grid_shape)


    def _get_shape(self):
        shape = {}
        for key, param in self.param_grid: 
            shape[key] = len(param)

        return shape

    def _get_next(self): 
        x = self.iterator()
        hyperparams = 

class GridSearchIterator: 
    def __init__(self, param_grid, grid_shape) -> None:

        self.param_grid = param_grid 
        self.current_indices = np.zeros(len(param_grid))
        self.shape = grid_shape
        
        self.num_elements = 0

    def __call__(self) -> None:
        
        for i , hpindex in enumerate(self.current_indices): 
            if hpindex != self.shape[hpindex] - 1: 
                self.current_indices[i] += 1
                break
            else: 
                self.current_indices[i] = 0
