
"""Lib specific type defs"""


from typing import Any, List, Dict
from dataclasses import dataclass


@dataclass
class Hyperparameter:
    """Base class for any hyperparameter"""
    val: Any
    
@dataclass
class ScalarHyperparameter(Hyperparameter):
    """For continuous/discrete scalar values like lr, momentum, dropout"""
    val: float
    lower: float
    upper: float
    logscale: bool = False  # for lr. makes search space logarithmic:  10^{-9} to 10^{-8} ~~ 1 to 2.
    
@dataclass 
class CategoricalHyperparameter(Hyperparameter):
    """For stuff like optimizer , activation function type """
    val: Any
    possiblevalues: List[Any]

# Type alias
ParameterSpace = Dict[str , Hyperparameter]
