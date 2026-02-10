
"""
Tinygrad's philosophy is one of simplicity and minimalism. As such,
it does not enforce ANY structure in the ways one wraps or declares a net
that will later be trained. Unlike PyTorch, which inherits from nn.Module

Thus, given that this is supplementary tooling to tinygrad, it will need 
some uniformity in its input. Either the user can do that or this lib,
so giving some tooling for wrapping tinygrad models is needed.
"""
from pyexpat import model

from dataclasses import dataclass
from typing import List, Callable


# to interact with the user's network
@dataclass
class hpo_callables:
    modelFit : Callable
    modelOptim : Callable
    modelLoss : Callable
    


def create_callables():