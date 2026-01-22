"""
Bayesian Optimization search strategy for hyperparameter tuning.

Uses a Gaussian Process surrogate model to model the objective function
and an acquisition function to balance exploration vs exploitation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

from .gaussianprocess import GaussianProcessRegressor
from .kernels import Kernel, LinearKernel, SquaredExponential, RBF, Matern32Kernel
from .acquisition import get_acquisition_function, expected_improvement, probability_of_improvement, lower_confidence_bound
from src.types import ParameterSpace
from ..gridsearch import GridSearch
from ..randomsearch import RandomSearch

class BayesianSearch:
    """Bayesian Optimization for hyperparameter search."""

    def __init__(self,
                 covkernel : Kernel,
                 parameterspace: ParameterSpace,
                 v = 1e-6, # noise variance
                 searchstrategy : str = "bayesian"
                 acquisition: str = 'ei',
                 n_initial: int = 5,
                 seed: Optional[int] = None) -> None:
        """
        Initialize Bayesian search.

        Args:
            param_grid: Dict mapping parameter names to lists of possible values
            acquisition: Acquisition function name ('ei', 'pi', 'lcb')
            n_initial: Number of random initial samples before using GP
            seed: Random seed for reproducibility
        """
        self.param_grid = param_grid
        self.acq = acquisition
        self.nsteps = n_initial
        self.process = GaussianProcessRegressor(
            covkernel,
            searchstrategy,
            parameterspace,
            v
        )
        
    
    def __call__(self):
        
        # optimize for the acquisition function for x_m+1.
        
