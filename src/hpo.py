from typing import Any, Callable, Optional, list, tuple, union, optional, dict, callable, any
from tinygrad import tensor, nn, device, dtype, tensor

from strategies import get_hpo_strategy

import numpy as np

from searchimport Path
__all__ = [
        randomSearch,   # explorative -- faster, less likely to find global maxima
        gridSearch,     # exploitative -- slower, guranteed to find global maxima, but sometimes computationally infeasible. Best use case is a small network
        bayesianSearch  # balanced -- best for larger nns.
        ]


class hyperparameter_optimizer:

    # TODO: assign types to args
    # TODO: Figure out data loading
    def __init__(self,
                 model,
                 optimizer,
                 loss_function : Callable,
                 dataset = None,
                 data_split : Callable = None,
                 Xtrain : List[Any],
                 Ytrain : List[Any],
                 Xtest,
                 Ytest,
                 parameter_grid,
                 metric_to_maximize,
                 maximize: bool = True,
                 search_method: str = 'bayesian',
                 ) -> None:

        

        self.model = model
        self.optimizer = optimizer
        self.parameter_grid = parameter_grid
        
        # -- scoring --
        self.metric = metric_to_maximize if maximize == True else -metric_to_maximize
        self.best_score = float("-inf")
        self.best_params = None
        self.learning_process = []      # keeps track of study process

        # gridsearch, random, or BO
        self.search_method = get_hpo_strategy(search_method)

        # data 
    """
     * calculates best hyperparams. 
    """

    def hyperoptimize(self, n_trials: int = 50,
                      parameter_grid,
                      search_method="bayesian",

                      ) -> dict[str, float]:

        for i in range(n_trials):

            # dict type
            current_hyperparameters, m_id = self.search_method()        # should return hp[], i[]
            optimizer_hp = current_hyperparameters[]

            model = self.build_model(current_hyperparameters)
            m : int = 1


    def build_model(self):
        pass

    def _evaluate_model():
        pass

    def _suggest_next():
        pass

    def save():
        pass
