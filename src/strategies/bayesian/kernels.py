


import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Interface for Kernel Function Classes."""
    @abstractmethod
    def __call__(self, X1, X2):
        ...


class RBF(Kernel):
    """Radial Basis Function (RBF) kernel."""

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        """Compute the RBF kernel between X1 and X2."""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / self.length_scale**2 * sqdist)

class SquaredExponential(Kernel):
    
    def __init__(self, length_scale : float = 1.0):
        self.l = length_scale
        
    def __call__(self, X1 : np.ndarray, X2 : np.ndarray):
        diff = X1 - X2
        return np.exp(-0.5 * (np.sum(diff**2)) / (self.l**2))


class Matern52Kernel(Kernel):
    """Matern 5/2 kernel"""
    def __init__(self, length_scale : float =1.0, sigma : float =1.0):
        self.length_scale = length_scale
        self.sigma = sigma
    
    def __call__(self, X1 : np.ndarray, X2: np.ndarray):
        r = np.sqrt(np.sum((X1[:, None] - X2[None, :])**2, axis=2))
        scaled_r = np.sqrt(5) * r / self.length_scale
        return self.sigma**2 * (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)

class Matern32Kernel(Kernel):
    """Matern 3/2 kernel"""
    def __init__(self, length_scale=1.0, sigma=1.0):
        self.length_scale = length_scale
        self.sigma = sigma
    
    def __call__(self, X1, X2):
        r = np.sqrt(np.sum((X1[:, None] - X2[None, :])**2, axis=2))
        scaled_r = np.sqrt(3) * r / self.length_scale
        return self.sigma**2 * (1 + scaled_r) * np.exp(-scaled_r)

class RationalQuadraticKernel(Kernel):
    """Rational quadratic kernel"""
    def __init__(self, length_scale=1.0, sigma=1.0, alpha=1.0):
        self.length_scale = length_scale
        self.sigma = sigma
        self.alpha = alpha
    
    def __call__(self, X1, X2):
        r_sq = np.sum((X1[:, None] - X2[None, :])**2, axis=2)
        return self.sigma**2 * (1 + r_sq / (2 * self.alpha * self.length_scale**2))**(-self.alpha)

class PeriodicKernel(Kernel):
    """Periodic kernel"""
    def __init__(self, length_scale=1.0, sigma=1.0, period=1.0):
        self.length_scale = length_scale
        self.sigma = sigma
        self.period = period
    
    def __call__(self, X1, X2):
        r = np.sqrt(np.sum((X1[:, None] - X2[None, :])**2, axis=2))
        sin_term = np.sin(np.pi * r / self.period)
        return self.sigma**2 * np.exp(-2 * sin_term**2 / self.length_scale**2)

class LinearKernel(Kernel):
    """Linear kernel"""
    def __init__(self, sigma_b=1.0, sigma_v=1.0, offset=0.0):
        self.sigma_b = sigma_b
        self.sigma_v = sigma_v
        self.offset = offset
    
    def __call__(self, X1, X2):
        return self.sigma_b**2 + self.sigma_v**2 * (X1 - self.offset) @ (X2 - self.offset).T

class WhiteNoiseKernel(Kernel):
    """White noise kernel (diagonal only)"""
    def __init__(self, sigma=1.0):
        self.sigma = sigma
    
    def __call__(self, X1, X2):
        if X1.shape == X2.shape and np.allclose(X1, X2):
            return self.sigma**2 * np.eye(X1.shape[0])
        return np.zeros((X1.shape[0], X2.shape[0]))