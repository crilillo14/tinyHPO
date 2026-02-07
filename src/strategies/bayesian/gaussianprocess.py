"""
            -- gaussianprocess.py --

Gaussian Process Regressor for Bayesian Optimization.
"""

import numpy as np
from scipy.optimize import minimize
from src.types import ParameterSpace, ScalarHyperparameter
from src.strategies.bayesian.kernels import Kernel, Matern32Kernel 

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.linalg import cho_solve


class GaussianProcess:
    def __init__(self,
                 covkernel : Kernel,
                 parameterspace : ParameterSpace,
                 acqfn : str = 'ei',
                 noise : float = 1e-6, 
                 ):
        
        self.kernel = covkernel
        self._acqfn = acqfn
        self.H = parameterspace
        self.parameterspace = parameterspace
        self.hparams = parameterspace.keys()
        self.R_h = len(parameterspace)
        self.bestguess : float = None 
        self.noise = noise
        
        
    def sample(self, n=50): 
        H_T : np.ndarray = np.array([h(n) for name, h in self.parameterspace.items()])
        return H_T.T
        

    def fit(self, X, y):
        self.X = X
        self.y = y

        K = self.kernel(X, X)
        K += (self.noise + 1e-8) * np.eye(len(X))

        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(
            self.L.T, np.linalg.solve(self.L, y)
        )

    def predict(self, X_star):
        Kxs = self.kernel(X_star, self.X)
        mu = Kxs @ self.alpha

        v = np.linalg.solve(self.L, Kxs.T)
        Kss = self.kernel(X_star, X_star)
        var = np.diag(Kss) - np.sum(v ** 2, axis=0)

        return mu, np.maximum(var, 1e-9)
    
    


        
        
        
        
        
        
        
        
        
        
def testsampling():
    k = Matern32Kernel()
    lr = ScalarHyperparameter(1e-6, 1e-10, 1e-5, 10, True)
    m = ScalarHyperparameter(0.9, 0.6, 0.99, 10)
    
    space : ParameterSpace = { "lr" : lr,
                     "m" : m    }
    
    gp = GaussianProcess(k, space)
    

    samples = gp.sample(n=50)
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
 
    sns.scatterplot(
        x=samples[:, 0],
        y=samples[:, 1],
        hue=samples[:, 1],
        palette="viridis",
        ax=ax_lin,
        legend=False,
    )
    ax_lin.set_xlabel("lr")
    ax_lin.set_ylabel("m")
    ax_lin.set_title("lr (linear)")
 
    sns.scatterplot(
        x=samples[:, 0],
        y=samples[:, 1],
        hue=samples[:, 1],
        palette="viridis",
        ax=ax_log,
        legend=False,
    )
    ax_log.set_xscale("log")
    ax_log.set_xlabel("lr")
    ax_log.set_title("lr (log)")
 
    plt.tight_layout()
    plt.show()
    
    
    
     
        

if __name__ == "__main__":
    
    # test sampling
    testsampling()