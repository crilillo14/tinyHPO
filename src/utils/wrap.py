
"""
Tinygrad's philosophy is one of simplicity and minimalism. As such,
it does not enforce ANY structure in the ways one wraps or declares a net
that will later be trained.

Thus, given that this is supplementary tooling to tinygrad, it will need 
some uniformity in its input. Either the user can do that or this lib,
so giving some tooling for wrapping tinygrad models is needed.
"""

from dataclasses import dataclass
from typing import List, Callable

@dataclass
class hpo_callables:
    modelFit : Callable
    


def create_callables()