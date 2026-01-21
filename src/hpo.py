from typing import Any, Callable, Optional, list, tuple, union, optional, dict, callable, any
from tinygrad import tensor, nn, device, dtype, tensor
import tinygrad

from strategies import get_hpo_strategy

import numpy as np

from typing import Any

"""
Search methods.

randomSearch: [explorative] -- faster, less likely to find global maxima 
gridSearch: [exploitative] -- slower, guranteed to find global maxima, but sometimes computationally infeasible. Best for small nets. 
bayesianSearch  # exploration-exploitation tradeoff -- best for high dimensional hyperparam spaces. 
"""


class HPOptimizer:

    # TODO: assign types to args
    # TODO: Figure out data loading
    def __init__(self,
                 model : Any, 
                 loss_function : Callable,
                 metric_to_maximize : Callable, # in most cases, an inference step followed by accuracy measurement. Can be loss too.
                 dataset = None,
                 maximize: bool = True,
                 search_method: str = 'bayesian',
                 ) -> None:

        

        self.model = model
        self.parameter_grid = parameter_grid
        self.X
        
        # -- scoring --
        self.metric = metric_to_maximize if maximize == True else -metric_to_maximize
        self.best_score = float("-inf")
        self.best_params = None
        self.hp_history = []      # keeps track of search history

        # gridsearch, random, or BO
        self.search_method = get_hpo_strategy(search_method)

        # data 
    """
     * calculates best hyperparams. 
    """

    def fit(self, n_trials: int = 50,
                      parameter_grid,
                      search_method="bayesian",

                      ) -> dict[str, float]:

        for i in range(n_trials):

            # dict type
            current_hyperparameters, m_id = self.search_method()        # should return hp[], i[]
            optimizer_hp = current_hyperparameters[]

            model = self.build_model(current_hyperparameters)
            m : int = 1


    def buildmodel(self):
        pass

    def fit():
        pass

    def __next__():
        """Next prediction. """
        pass

    def saveconfig():
        """Save hyperparameter values"""
        pass
