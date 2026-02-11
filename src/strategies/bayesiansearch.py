
"""
Search Type wrapper around gaussian process. Owns the surrogate model, runs the search.
"""

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
    def __init__(
        self,
        parameterspace: ParameterSpace,
        objective_fn,
        n_initial: int = 10,
        n_iterations: int = 50,
        acqfn: str = "ei",
        noise: float = 1e-6,
        n_restarts: int = 5,
        minimize_objective: bool = True,
    ):
        self.parameterspace = parameterspace
        self.objective_fn = objective_fn
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self.minimize_objective = minimize_objective

        # Surrogate — only does fit() and predict()
        self.gp = GaussianProcess(
            covkernel=Matern32Kernel(),
            parameterspace=parameterspace,
            noise=noise,
        )

        # Acquisition — owned by the search, not the GP
        self.acqfn = get_acquisition_function(acqfn)

        self.X_observed = None
        self.y_observed = None
        self.best_config = None
        self.best_score = None
        self.history = []

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self):
        self._initialize()

        for i in range(self.n_iterations):
            # Surrogate just fits and predicts — that's it
            self.gp.fit(self.X_observed, self.y_observed)

            # Search owns acquisition optimization
            x_next = self._optimize_acquisition()

            config = self._array_to_config(x_next)
            score = self.objective_fn(config)

            self.X_observed = np.vstack([self.X_observed, x_next.reshape(1, -1)])
            self.y_observed = np.append(self.y_observed, score)

            self._update_best(config, score)
            self.history.append((config.copy(), score))

        return self.best_config, self.best_score

    # ------------------------------------------------------------------ #
    #  Acquisition — fully owned by the search                           #
    # ------------------------------------------------------------------ #

    def _optimize_acquisition(self) -> np.ndarray:
        bounds = self._get_bounds()
        best_x, best_acq = None, -np.inf

        for _ in range(self.n_restarts):
            x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])

            result = minimize(
                fun=lambda x: -self._acquisition_at(x),
                x0=x0,
                bounds=bounds,
                method="L-BFGS-B",
            )

            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x

        return best_x

    def _acquisition_at(self, x: np.ndarray) -> float:
        """
        Search asks surrogate for (mu, var).
        Search evaluates acquisition itself.
        Clean boundary: GP never sees the acquisition function.
        """
        x = x.reshape(1, -1)
        mu, var = self.gp.predict(x)
        return self.acqfn(
            mu=mu[0],
            var=var[0],
            best=self._current_best(),
        )

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _initialize(self):
        X_init = self.gp.sample(n=self.n_initial)
        y_init = np.array([
            self.objective_fn(self._array_to_config(x)) for x in X_init
        ])

        self.X_observed = X_init
        self.y_observed = y_init

        for x, y in zip(X_init, y_init):
            cfg = self._array_to_config(x)
            self._update_best(cfg, y)
            self.history.append((cfg.copy(), y))

    def _current_best(self) -> float:
        if self.minimize_objective:
            return np.min(self.y_observed)
        return np.max(self.y_observed)

    def _update_best(self, config: dict, score: float):
        if self.best_score is None:
            self.best_config, self.best_score = config, score
            return
        if self.minimize_objective and score < self.best_score:
            self.best_config, self.best_score = config, score
        elif not self.minimize_objective and score > self.best_score:
            self.best_config, self.best_score = config, score

    def _get_bounds(self):
        return [(h.low, h.high) for _, h in self.parameterspace.items()]

    def _array_to_config(self, x: np.ndarray) -> dict:
        return {
            name: float(x[i])
            for i, name in enumerate(self.parameterspace.keys())
        }