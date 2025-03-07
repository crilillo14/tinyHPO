import math
import random
from typing import Optional

class GaussianProcess:

    """ Gaussian Process { X1, X2, ... , Xn } ~ N(mean , variance).
    """

    def __init__(self,
                 length,
                 mean,
                 variance,
                 isStochastic : Optional[bool]
                 ):
        
        self.length = length
        self.isStochastic = isStochastic
        self.process = self.initiateProcess(self.length)
        self.mu = mean
        self.var = variance
       
        
    def generateProcess(self): 
        self.process = [self._calculateRV() for x in range(self.length)]
        return [self._calculateRV() for x in range(self.length)]
    
    def _calculateRV(self):
        # get a random val with ~ N(0 , 1)
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)  
        return z0*self.var + self.mu