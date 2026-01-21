
"""Lib specific type defs"""

from typing import Any, List
from dataclasses import dataclass
from numbers import Real

@dataclass
class Hyperparameter:
    """Base class for any hyperparameter"""
    name: str
    val: Any
    
@dataclass
class ScalarHyperparameter(Hyperparameter):
    """For continuous/discrete scalar values like lr, momentum, dropout"""
    val: Real
    lower: Real
    upper: Real
    logscale: bool = False  # for lr. makes search space logarithmic:  10^{-9} to 10^{-8} ~~ 1 to 2.
    
@dataclass 
class CategoricalHyperparameter(Hyperparameter):
    """For stuff like optimizer , activation function type """
    val: Any
    possiblevalues: List[Any]

# Type alias
ParameterSpace = List[Hyperparameter]
