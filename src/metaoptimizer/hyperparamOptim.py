# imports 


from typing import List, Callable, Type, Tuple, Dict, Optional, Union
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import itertools
import random
from dataclasses import dataclass

x = Tensor([[1, 2]])
print (x)

@dataclass
class OptimizationResult:
    """Stores the results of hyperparameter optimization"""
    best_params: Dict[str, Union[int, float]]
    best_score: float
    all_results: List[Tuple[Dict[str, Union[int, float]], float]]

class HyperparameterOptimizer:
    """A general purpose hyperparameter optimizer for tinygrad models."""
    
    def __init__(self,
                 model_class: Type,
                 train_data: Tuple[Tensor, Tensor],
                 hyperparam_space: Dict[str, Tuple[float, float]],
                 val_split: float = 0.2,
                 metric: Callable = None,
                 minimize: bool = True):
        """
        Args:
            model_class: The class of the model to optimize
            train_data: Tuple of (features, labels) tensors
            hyperparam_space: Dict of parameter names and their (min, max) bounds
            val_split: Fraction of data to use for validation
            metric: Custom evaluation metric (defaults to MSE)
            minimize: Whether to minimize (True) or maximize (False) the metric
        """
        self.model_class = model_class
        self.hyperparam_space = hyperparam_space
        self.minimize = minimize
        self.metric = metric if metric else self._default_metric
        
        # Split data into train and validation sets
        self.train_data, self.val_data = self._split_train_val_tensors(train_data, val_split)
        
    def _split_train_val_tensors(self, 
                                data: Tuple[Tensor, Tensor], 
                                val_split: float) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Splits data into training and validation sets"""
        x, y = data
        n = x.shape[0]
        idx = int(n * (1 - val_split))
        
        train_x, val_x = x[:idx], x[idx:]
        train_y, val_y = y[:idx], y[idx:]
        
        return (train_x, train_y), (val_x, val_y)
    
    def _default_metric(self, y_true: Tensor, y_pred: Tensor) -> float:
        """Default evaluation metric (MSE)"""
        return float(((y_true - y_pred) ** 2).mean().numpy())
    
    def _evaluate_model(self, params: Dict[str, Union[int, float]]) -> float:
        """Trains and evaluates a model with given parameters"""
        try:
            # Instantiate model with parameters
            model = self.model_class(**params)
            
            # Train model (assuming model has a train method)
            model.train(self.train_data[0], self.train_data[1])
            
            # Evaluate on validation set
            with Tensor.no_grad():
                val_pred = model(self.val_data[0])
                score = self.metric(self.val_data[1], val_pred)
            
            return score
        except Exception as e:
            print(f"Error evaluating parameters {params}: {str(e)}")
            return float('inf') if self.minimize else float('-inf')

    def grid_search(self, num_points: int = 5) -> OptimizationResult:
        """Performs grid search over the parameter space"""
        # Create grid points for each parameter
        param_grid = {}
        for param, (min_val, max_val) in self.hyperparam_space.items():
            param_grid[param] = np.linspace(min_val, max_val, num_points)
        
        # Generate all combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        # Evaluate all combinations
        results = []
        for params in param_combinations:
            score = self._evaluate_model(params)
            results.append((params, score))
        
        # Find best result
        best_idx = min(range(len(results)), key=lambda i: results[i][1]) if self.minimize \
                   else max(range(len(results)), key=lambda i: results[i][1])
        
        return OptimizationResult(
            best_params=results[best_idx][0],
            best_score=results[best_idx][1],
            all_results=results
        )
    
    def bayesian_optimize(self, n_iterations: int = 50) -> OptimizationResult:
        """Performs Bayesian optimization using Gaussian Process"""
        # Initialize GP
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        # Initial random points
        n_initial = min(5, n_iterations)
        X_sample = []
        y_sample = []
        results = []
        
        # Helper to convert params to/from array
        def params_to_array(params):
            return np.array([params[k] for k in self.hyperparam_space.keys()])
        
        def array_to_params(arr):
            return dict(zip(self.hyperparam_space.keys(), arr))
        
        # Initial random sampling
        for _ in range(n_initial):
            params = {k: random.uniform(v[0], v[1]) 
                     for k, v in self.hyperparam_space.items()}
            score = self._evaluate_model(params)
            X_sample.append(params_to_array(params))
            y_sample.append(score)
            results.append((params, score))
        
        # Main optimization loop
        for i in range(n_initial, n_iterations):
            X = np.array(X_sample)
            y = np.array(y_sample)
            
            # Fit GP
            gp.fit(X, y)
            
            # Find next point to evaluate using expected improvement
            best_y = min(y) if self.minimize else max(y)
            
            def objective(x):
                x = x.reshape(1, -1)
                mean, std = gp.predict(x, return_std=True)
                if self.minimize:
                    z = (best_y - mean) / std
                    ei = std * (z * norm.cdf(z) + norm.pdf(z))
                else:
                    z = (mean - best_y) / std
                    ei = std * (z * norm.cdf(z) + norm.pdf(z))
                return -ei
            
            # Random search for next point
            best_ei = float('inf')
            best_params = None
            
            for _ in range(100):
                x = np.array([random.uniform(v[0], v[1]) 
                            for v in self.hyperparam_space.values()])
                ei = objective(x)
                if ei < best_ei:
                    best_ei = ei
                    best_params = x
            
            # Evaluate new point
            params = array_to_params(best_params)
            score = self._evaluate_model(params)
            
            X_sample.append(best_params)
            y_sample.append(score)
            results.append((params, score))
        
        # Find best result
        best_idx = min(range(len(results)), key=lambda i: results[i][1]) if self.minimize \
                   else max(range(len(results)), key=lambda i: results[i][1])
        
        return OptimizationResult(
            best_params=results[best_idx][0],
            best_score=results[best_idx][1],
            all_results=results
        )
