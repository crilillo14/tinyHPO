from typing import list, tuple, union, optional, dict, callable, any
from tinygrad import tensor, nn, device, dtype, tensor

from strategies import get_hpo_strategy

import numpy as np


class hyperparameter_optimizer:


    # TODO: assign types to args
    def __init__(self, 
                 model,
                 param_grid,
                 metric_to_maximize,
                 maximize : bool = True,
                 search_method : str = 'bayesian',
                 ) -> None:

        
        # props

        self.model = model
        self.metric = metric_to_maximize if maximize == True else -metric_to_maximize
        self.best_score = float("-inf")
        self.best_params = None
        self.learning_process = []      # keeps track of study process
    
        # gridsearch, random, or BO
        self.search_method = get_hpo_strategy(search_method)


        

    """
     * calculates best hyperparams. 
    """

    def study(self, n_trials: int = 50, 
              param_ranges: dict[str, tuple[float, float]] = None,
              optimizer_method = "bayesian",

              ) -> dict[str, float]:
 
         """find optimal values for hyperparameters 

        returns:
            dict[str , float]: hyperparameter names with corresponding values. 
        """
        
        hp_optimizer = 
       
        



    def build_model(self): 
        pass

    
    def _evaluate_model() :
        pass

    def _suggest_next(): 
        pass
    
    def save():
        pass 


