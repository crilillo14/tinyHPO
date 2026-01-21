"""
            -- gaussianprocess.py --

Gaussian Process Regressor for Bayesian Optimization.
"""

import numpy as np
from scipy.optimize import minimize


class Kernel:
    """Base class for kernel functions."""
    def __call__(self, X1, X2):
        raise NotImplementedError


class RBF(Kernel):
    """Radial Basis Function (RBF) kernel."""

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        """Compute the RBF kernel between X1 and X2."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / self.length_scale**2 * sqdist)


class GaussianProcessRegressor:
    """Gaussian Process Regressor for modeling performance as a function of hyperparameters."""

    def __init__(self, kernel, noise=1e-10):
        self.kernel = kernel
        self.noise = noise

    def log_marginal_likelihood(self, length_scale):
        """Compute the negative log marginal likelihood for optimization."""
        self.kernel.length_scale = length_scale
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        log_likelihood = -0.5 * self.y_train.T @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(self.X_train) * np.log(2 * np.pi)
        return -log_likelihood  # Minimize negative likelihood

    def fit(self, X, y):
        """
        Fit the Gaussian process to the training data.

        Args:
            X (np.ndarray): Training inputs (hyperparameters), shape (n_samples, n_features).
            y (np.ndarray): Training outputs (performances), shape (n_samples,).
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        # Optimize kernel length_scale
        res = minimize(self.log_marginal_likelihood, [1.0], bounds=[(0.1, 10.0)])
        self.kernel.length_scale = res.x[0]

        # Precompute for prediction
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))

    def predict(self, X_new, return_std=False):
        """
        Predict mean and optionally standard deviation for new points.

        Args:
            X_new (np.ndarray): New inputs, shape (n_samples, n_features).
            return_std (bool): If True, return standard deviation.

        Returns:
            np.ndarray: Predicted means.
            np.ndarray (optional): Standard deviations if return_std is True.
        """
        X_new = np.array(X_new)
        K_s = self.kernel(self.X_train, X_new)
        mu = K_s.T @ self.alpha

        if return_std:
            K_ss = self.kernel(X_new, X_new)
            v = np.linalg.solve(self.L, K_s)
            var = K_ss - v.T @ v
            return mu, np.sqrt(np.maximum(np.diag(var), 0))  # Ensure non-negative variance

        return mu
