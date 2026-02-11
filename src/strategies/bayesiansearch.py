
"""
Search Type wrapper around gaussian process. Owns the surrogate model, runs the search.
"""
from strategies.bayesian.kernels import Kernel, SquaredExponential


from .bayesian.gaussianprocess import GaussianProcess
from src.search import SearchStrategy





import numpy as np
from scipy.optimize import minimize

from src.strategies.bayesian.gaussianprocess import GaussianProcess
from src.strategies.bayesian.kernels import Matern32Kernel
from src.strategies.bayesian.acquisition import get_acquisition_function
from src.types import ParameterSpace


"""
    -- bayesian_search.py --

Bayesian Optimization loop. Owns the strategy.
The GP is just a surrogate it queries — nothing more.
"""

class BayesianSearch(SearchStrategy):
    def __init__(self, parameterspace, n_iterations, acqfn="ei", n_initial=10, kernel : Kernel = SquaredExponential, **kwargs):
        super().__init__(parameterspace, n_iterations)
        self.n_initial = n_initial
        self.gp = GaussianProcess(covkernel=Matern32Kernel(), parameterspace=parameterspace)
        self.acqfn = get_acquisition_function(acqfn)
        self.X_observed = []
        self.y_observed = []

    def suggest(self) -> Dict[str, Any]:
        # Warm-up: random points
        if self._iteration < self.n_initial:
            x = self.gp.sample(n=1)[0]
            return self._array_to_config(x)

        # After warm-up: fit GP, optimize acquisition
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        x_next = self._optimize_acquisition()
        return self._array_to_config(x_next)

    def update(self, config: Dict[str, Any], score: float) -> None:
        self.X_observed.append(self._config_to_array(config))
        self.y_observed.append(score)
        self._iteration += 1

        if self._best_score is None or score < self._best_score:
            self._best_config = config
            self._best_score = score

    # ... private methods (_optimize_acquisition, etc.)

