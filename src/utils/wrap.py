
"""
Tinygrad's philosophy is one of simplicity and minimalism. As such,
it does not enforce ANY structure in the ways one wraps or declares a net
that will later be trained. Unlike PyTorch, which inherits from nn.Module

Thus, given that this is supplementary tooling to tinygrad, it will need 
some uniformity in its input. Either the user can do that or this lib,
so giving some tooling for wrapping tinygrad models is needed.
"""
from src.types import ParameterSpace, ScalarHyperparameter, CategoricalHyperparameter
from src.search import get_hpo_strategy
import random


from dataclasses import dataclass
from typing import List, Callable, Union, Tuple





"""

Config can pass one of three types representing a linear / loglinear space of values:
    1. A list of values
    2. A tuple of (min, max, num_samples)
    3. an iterator type

Apart from that, it can alos pass a callable, and dependency inject its data as kwargs.


like you could do 

config = SearchSpaceConfig()
    .set_search(GridSearch)
    .append_hyperparameter("learning_rate", (0.001, 0.1, 5))
    .append_hyperparameter("batch_size", (16, 128, 5))
    .append_hyperparameter("dropout", [0.1, 0.5, 0.9], default=0.5)
    .set_iterations(50)


if given a list of values, search space is discrete.
if given a range, search space is continuous.



"""

type SingletonSpace = Union[List[float], Tuple[float, float, int]]

class SearchSpaceConfig():
    def __init__(self, **kwargs):
        self.hyperparameters : ParameterSpace = {}
        self.iterations : int = 50
        self.search_method : str = "random"
        self.acquisition_function : str = "EI"
        for key, value in kwargs.items():
            setattr(self, key, value)
                
    def set_search(self, search : str, acquisition_function="EI"):
        self.search_method = search
        if search == "bayesian":
            self.acquisition_function = acquisition_function
        return self
    
    def set_iterations(self, iterations : int) -> SearchSpaceConfig:
        self.iterations = iterations
        return self
    
    
    def append_hyperparameter(self, name: str, space: SingletonSpace, default=None, logscale=False) -> SearchSpaceConfig:
        if isinstance(space, tuple):
            low, high, num_samples = space
            self.hyperparameters[name] = ScalarHyperparameter(
                val=default or low,
                lower=low,
                upper=high,
                partitions=num_samples,
                logscale=logscale
            )
        elif isinstance(space, list):
            raise NotImplementedError("Discrete search spaces are not yet supported")
        else:
            raise ValueError("Invalid search space")
        return self
