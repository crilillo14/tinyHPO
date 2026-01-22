"""
            -- gaussianprocess.py --

Gaussian Process Regressor for Bayesian Optimization.
"""

import numpy as np
from scipy.optimize import minimize
from src.types import ParameterSpace
from src.strategies.bayesian.kernels import * 

class GaussianProcessRegressor:
    """Regresses on best guess given acq fn"""

    def __init__(self,
                 covkernel : Kernel,
                 searchstrategy : str,
                 parameterspace : ParameterSpace,
                 noise : float = 1e-6,
                 
                 ):
        pass
    
    def __call__(self):
        pass
    
    