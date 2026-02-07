
"""
Search Type wrapper around gaussian process. Really no additional logic here.
"""

from .bayesian.gaussianprocess import GaussianProcess

class BayesianSearch:
    
    def __init__(self, gp : GaussianProcess
                    ):
        self.gp = gp
        
    # Don't want to go per step, run full GP trial and then return all params
    # if user wants to step, keep surrogate process history in memory 
    # and continue from there
    
    def search(self):
        pass
    
    def step(self):
        pass