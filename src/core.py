from typing import List, Tuple, Union, Optional, Dict, Callable, Any
from tinygrad import Tensor, nn, device, dtype, tensor

from utils.gaussianProcess import GaussianProcess

import numpy as np


class hyperparameterOptimizer:

    def __init__(self, model, modelType, metricToMaximize) -> None:

        
        # props

        self.model = model
        self.metric = metricToMaximize
        self.gp = GaussianProcess()
        self.bestScore = float("-inf")
        self.bestParams = None
        self.learningProcess = []      # keeps track of study process
        

    """
     * calculates best hyperparams. 
    """

    def study(self, n_trials: int = 50, param_ranges: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
 
         """find optimal values for hyperparameters 

        Returns:
            Dict[str , float]: hyperparameter names with corresponding values. 
        """
        
        pass



    def buildModel(self): 
        pass

    
    def _evaluateModel() :
        pass

    def _suggestNext(): 
        pass
    
    def save():
        pass 