from typing import Any, Callable, Optional, List, Tuple, Union, Optional, Dict, Callable, Any
from tinygrad import tensor, nn, device, dtype, tensor
import tinygrad

from search import get_hpo_strategy
from src.strategies.bayesian.gaussianprocess import GaussianProcessRegressor
import numpy as np

from typing import Any
from src.types import ParameterSpace

from abc import ABC, abstractmethod

"""
Search methods.

randomSearch: [explorative] -- faster, less likely to find global maxima 
gridSearch: [exploitative] -- slower, guranteed to find global maxima, but sometimes computationally infeasible. Best for small nets. 
bayesianSearch  # exploration-exploitation tradeoff -- best for high dimensional hyperparam spaces. 
"""


# in general, an hpo just needs to know methods for inference and measure f(x), and a parameter space.
# Canonicalizing 


class HPO(ABC):
    # Should: 
    # contain a search object
    # hooks to wandb, viz search history, interface for moving around search.

    @abstractmethod
    def visualize(self):
        pass
    
    @abstractmethod
    def __iter__(self):
        """Returns an iterator object of search historsearch historyy. 
        """
        pass
class HyperparameterOptimizer(HPO):

    def __init__(self,
                 model : Any,
                 optimizer : Any,
                 loss_function : Callable,
                 metric_to_minimize : Callable, # in most cases, an inference step followed by loss measurement
                 parameterspace : ParameterSpace,
                 dataset,
                 maximize: bool = False,
                 search_method: str = 'bayesian',
                 acq_fn : str = 'ei'
                 ) -> None:

        

        self.model = model
        self.pspace = parameterspace
        self.data = dataset

        # -- scoring --
        self.maximizing = maximize
        self.best_score = float("-inf")
        self.best_params = None
        self.hp_history = []      # keeps track of search history

        # gridsearch, random, or bool

        self.search = get_hpo_strategy()

            

    """
     * calculates best hyperparams. 
    """

    def fit(self):
    
    def model(self):
        pass

    def bayesianFit(self):
        
        """def bayes_opt_step(gp, data, space):
    candidates = sample(space)
    mu, sigma = gp.predict(candidates)
    scores = EI(mu, sigma)
    return candidates[argmax(scores)]""" 


    def __next__():
        """Next prediction. """
        pass

    def saveconfig():
        """Save hyperparameter values"""
        pass
