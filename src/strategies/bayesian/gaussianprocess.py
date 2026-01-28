"""
            -- gaussianprocess.py --

Gaussian Process Regressor for Bayesian Optimization.
"""

import numpy as np
from scipy.optimize import minimize
from src.types import ParameterSpace
from src.strategies.bayesian.kernels import Kernel 

from scipy.linalg import cho_solve

class GaussianProcessRegressor:
    """Regresses on best guess given acq fn"""

    def __init__(self,
                 covkernel : str,
                 parameterspace : ParameterSpace,
                 acqfn : str = 'ei',
                 noise : float = 1e-6, 
                 ):
        
        self.kernel = self._initiateCovKernel(covkernel, noise)
        self._acqfn = acqfn
        self.H = parameterspace
        self.means = 
        
        
        
    def __call__(self):
        pass


    def _initiateCovKernel(self, kernelName : str, noise):    
        pass
    
    def callkernel(self, x):
        pass
    
    def callmean(self, x):
        pass
    
    def fit(self, x): 
        pass
    
    def choleskyDecomposition(self, X):
        pass
        
    def marginalLogLikelihood(self, X):
       pass  
        