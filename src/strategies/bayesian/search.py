"""
            -- bayesian/search.py --

Bayesian Optimization search strategy for hyperparameter tuning.

Uses a Gaussian Process surrogate model to model the objective function
and an acquisition function to balance exploration vs exploitation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

from .gaussianprocess import GaussianProcessRegressor, RBF
from .acquisition import get_acquisition_function, expected_improvement


class BayesianSearch:
    """Bayesian Optimization for hyperparameter search."""

    def __init__(self,
                 param_grid: Dict[str, List[Any]],
                 acquisition: str = 'ei',
                 n_initial: int = 5,
                 seed: Optional[int] = None) -> None:
        """
        Initialize Bayesian search.

        Args:
            param_grid: Dict mapping parameter names to lists of possible values
            acquisition: Acquisition function name ('ei', 'ucb', 'pi', 'lcb')
            n_initial: Number of random initial samples before using GP
            seed: Random seed for reproducibility
        """
        self.param_grid = param_grid
        self.keys = list(param_grid.keys())
        self.values = list(param_grid.values())

        # Build candidate space (all possible combinations)
        self.candidates = self._build_candidate_space()
        self.candidate_indices = {tuple(c): i for i, c in enumerate(self.candidates)}

        # Gaussian Process surrogate
        self.gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), noise=1e-6)

        # Acquisition function
        self.acquisition_name = acquisition
        self.acquisition_fn = get_acquisition_function(acquisition)

        # State
        self.n_initial = n_initial
        self.rng = np.random.default_rng(seed)
        self.X_observed = []  # Observed parameter vectors
        self.y_observed = []  # Observed scores
        self.history = []     # History of parameter dicts
        self.iteration = 0

    def _build_candidate_space(self) -> np.ndarray:
        """Build the full candidate space as encoded vectors."""
        # For discrete param_grid, we enumerate all combinations
        all_combos = list(product(*[range(len(v)) for v in self.values]))
        return np.array(all_combos, dtype=float)

    def _encode_params(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameter dict to numeric vector."""
        encoded = []
        for key in self.keys:
            val = params[key]
            idx = self.values[self.keys.index(key)].index(val)
            encoded.append(idx)
        return np.array(encoded, dtype=float)

    def _decode_params(self, encoded: np.ndarray) -> Dict[str, Any]:
        """Decode numeric vector back to parameter dict."""
        params = {}
        for i, key in enumerate(self.keys):
            idx = int(encoded[i])
            params[key] = self.values[i][idx]
        return params

    def update(self, params: Dict[str, Any], score: float) -> None:
        """
        Update the surrogate model with a new observation.

        Args:
            params: The parameter dict that was evaluated
            score: The score/metric from evaluating those parameters
        """
        x = self._encode_params(params)
        self.X_observed.append(x)
        self.y_observed.append(score)

    def __call__(self) -> Dict[str, Any]:
        """
        Suggest the next hyperparameter set to evaluate.

        Returns:
            Dict of hyperparameters to try next
        """
        self.iteration += 1

        # Initial random sampling phase
        if len(self.X_observed) < self.n_initial:
            return self._random_sample()

        # Fit GP on observed data
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        # Find candidates we haven't tried yet
        tried = set(tuple(x) for x in self.X_observed)
        untried_mask = [tuple(c) not in tried for c in self.candidates]
        untried_candidates = self.candidates[untried_mask]

        if len(untried_candidates) == 0:
            print("Bayesian search exhausted - all candidates tried")
            return None

        # Get acquisition values for untried candidates
        mu, sigma = self.gp.predict(untried_candidates, return_std=True)
        y_best = np.max(self.y_observed)

        # Compute acquisition values
        if self.acquisition_name == 'ei':
            acq_values = expected_improvement(mu, sigma, y_best)
        elif self.acquisition_name == 'ucb':
            acq_values = self.acquisition_fn(mu, sigma, kappa=2.0)
        elif self.acquisition_name == 'pi':
            acq_values = self.acquisition_fn(mu, sigma, y_best)
        elif self.acquisition_name == 'lcb':
            acq_values = self.acquisition_fn(mu, sigma, kappa=2.0)
        else:
            acq_values = self.acquisition_fn(mu, sigma)

        # Select candidate with highest acquisition value
        best_idx = np.argmax(acq_values)
        best_candidate = untried_candidates[best_idx]

        params = self._decode_params(best_candidate)
        self.history.append(params)
        return params

    def _random_sample(self) -> Dict[str, Any]:
        """Sample a random untried candidate."""
        tried = set(tuple(x) for x in self.X_observed)
        untried_mask = [tuple(c) not in tried for c in self.candidates]
        untried_candidates = self.candidates[untried_mask]

        if len(untried_candidates) == 0:
            return None

        idx = self.rng.integers(0, len(untried_candidates))
        params = self._decode_params(untried_candidates[idx])
        self.history.append(params)
        return params

    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """Return the best parameters and score observed so far."""
        if not self.y_observed:
            return None, None
        best_idx = np.argmax(self.y_observed)
        best_params = self._decode_params(self.X_observed[best_idx])
        return best_params, self.y_observed[best_idx]

    def reset(self) -> None:
        """Reset the search state."""
        self.X_observed = []
        self.y_observed = []
        self.history = []
        self.iteration = 0

    def __len__(self) -> int:
        """Return total number of possible combinations."""
        return len(self.candidates)

    def __repr__(self) -> str:
        return (f"BayesianSearch(params={self.keys}, "
                f"acquisition={self.acquisition_name}, "
                f"observed={len(self.X_observed)})")
