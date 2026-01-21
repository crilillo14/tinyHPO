
"""Lib specific type defs"""

from typing import Union, Dict, Tuple
from dataclasses import dataclass
from tinygrad import Tensor, dtypes

Real = Union[dtypes.float, dtypes.int , dtypes.double]

@dataclass
class Hyperparameter:
    val : Real
    parameterspace : Tensor
    
ParameterGrid = Dict[str, Tuple[Real]]

__all__ = [
    Real,
    Hyperparameter 
]