
"""Lib specific type defs"""


from typing import Any, List, Dict
from dataclasses import dataclass

import numpy as np


# ---------- Dataclasses ----------

@dataclass
class Hyperparameter:
    """Base class for any hyperparameter"""
    val: Any
    
    def __len__(self):
        raise NotImplementedError("len should be implemeneted in subclasses")
    def __call__(self, n):
        raise NotImplementedError("call should be implemeneted in subclasses")
    
@dataclass
class ScalarHyperparameter(Hyperparameter):
    """For continuous/discrete scalar values like lr, momentum, dropout"""
    val: float
    lower: float
    upper: float
    partitions : int
    logscale: bool = False  # for lr. makes search space logarithmic:  10^{-9} to 10^{-8} ~~ 1 to 2.
    
    def __len__(self):
        return self.partitions

    def __call__(self, n):
        if self.logscale:
            return 10 ** np.random.uniform(
                np.log10(self.lower),
                np.log10(self.upper),
                size=n
                )
        else:
            return np.random.uniform(
                low = self.lower,
                high = self.upper,
                size = n
            )
    
@dataclass 
class CategoricalHyperparameter(Hyperparameter):
    """For stuff like optimizer , activation function type """
    val: Any
    possiblevalues: List[Any]

    def __len__(self):
        return len(self.possiblevalues)

    def __call__(self, n):
        raise NotImplementedError("Hyperparameter Optimization for Non scalar hyperparameters has not yet been implemented.")


# ---------- Type aliases
ParameterSpace = Dict[str , Hyperparameter]
