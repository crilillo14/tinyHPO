
"""Lib specific type defs"""

from typing import Union, Dict, Tuple, Any
from dataclasses import dataclass
from tinygrad import Tensor, dtypes


Real = Union[dtypes.float, dtypes.int , dtypes.double]

@dataclass
class Hyperparameter:
    val : Real
    parameterspace : Tensor
    
ParameterGrid = Dict[str, Tuple[Real]]

class Config:
    model : Any     # design choice of tinygrad to have no interface for models. model could be a fn too.
    optimizer : Any
    
    



__all__ = [
    Real,
    Hyperparameter 
]