# reduced grid search with bayesian optim

from utils.gaussianProcess import GaussianProcess

class BayesianOptimization:
    
    def __init__(self, 
                 gaussian : GaussianProcess,
                 ) -> None:
         