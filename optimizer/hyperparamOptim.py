# imports 


from typing import List, Callable, Type, Tuple, Dict
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange

x = Tensor([[1, 2]])
print (x)

class hyperparameterOptimizer : 
    """ A general purpose hyperparameter optimizer for tinygrad models.
    """
    def __init__(self,
                 modelClass : Type,
                 trainData : Tuple[Tensor, Tensor], 
                 validationData: Tuple[Tensor, Tensor],
                 # 
                 hyperparamSpace : Dict[str , str]
                 
                 ):
        pass