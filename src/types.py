
"""Lib specific type defs"""


from typing import Any, List, Dict
from dataclasses import dataclass


# ---------- Dataclasses ----------

@dataclass
class Hyperparameter:
    """Base class for any hyperparameter"""
    val: Any
    
    def __len__(self):
        raise NotImplementedError("len should be implemeneted in subclasses")
    
@dataclass
class ScalarHyperparameter(Hyperparameter):
    """For continuous/discrete scalar values like lr, momentum, dropout"""
    val: float
    lower: float
    upper: float
    logscale: bool = False  # for lr. makes search space logarithmic:  10^{-9} to 10^{-8} ~~ 1 to 2.
    partitions : int
    
    def __len__(self):
        return self.partitions
    
@dataclass 
class CategoricalHyperparameter(Hyperparameter):
    """For stuff like optimizer , activation function type """
    val: Any
    possiblevalues: List[Any]

    def __len__(self):
        return len(self.possiblevalues)


# ---------- Type aliases
ParameterSpace = Dict[str , Hyperparameter]
