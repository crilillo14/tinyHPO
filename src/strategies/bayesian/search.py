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
    """Bayesian Optimization for hyperparameter search.""""

    def __init__( self,
                 covkernel : Kernel,
                 parameterspace: ParameterSpace,
                 X,
                 Y,
                 v = 1e-6, # noise variance
                 acquisition: str = 'ei',
                 steps: int = 5,
                 seed: Optional[int] = None
                 )  -> None:
        self.steps = steps
        
        self.param_grid = parameterspace 

        self.acq = acquisition
        self.nsteps = steps 
        self.process = GaussianProcessRegressor(
            covkernel,
            parameterspace,
            X
        )
        
        """
        Initialize Bayesian search.

        Args:
            param_grid: Dict mapping parameter names to lists of possible values
            acquisition: Acquisition function name ('ei', 'pi', 'lcb')
            n_initial: Number of random initial samp/les before using GP
            seed: Random seed for reproducibility
        """
    
    def __call__(self):
        
        # optimize for the acquisition function for x_m+1.
        
        
        pass
        

    def sample(self):
        """ Get next x"""
