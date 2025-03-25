import numpy as np

from sklearn.gaussian_process import gaussian_process_regressor
from sklearn.gaussian_process.kernels import matern


class gaussian_process:
    """wrapper around sklearn's gaussian_process_regressor for bayesian optimization"""
    
    def __init__(self, kernel=none, alpha=1e-6, n_restarts_optimizer=5):
        """initialize gaussian process.
        
        args:
            kernel: covariance function. default is matern kernel.
            alpha: value added to diagonal of covariance matrix. defaults to 1e-6.
            n_restarts_optimizer: number of restarts for optimizer. defaults to 5.
        """
        if kernel is None:
            # matern kernel with nu=2.5 (differentiable twice)
            kernel = matern(nu=2.5)
        
        self.gp = gaussian_process_regressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=true,
            random_state=42
        )
        self.x_train = none
        self.y_train = none
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> none:
        """fit the gaussian process model.
        
        args:
            x: array of hyperparameter configurations
            y: array of corresponding performance metrics
        """
        self.x_train = x
        self.y_train = y
        self.gp.fit(x, y)
        
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """make predictions with the gaussian process.
        
        args:
            x: points at which to predict
            
        returns:
            tuple of (mean predictions, standard deviations)
        """
        if self.x_train is none:
            return np.zeros(x.shape[0]), np.ones(x.shape[0])
        
        mean, std = self.gp.predict(x, return_std=true)
        return mean, std
        
    def update(self, x_new: np.ndarray, y_new: np.ndarray) -> none:
        """update the gaussian process with new observations.
        
        args:
            x_new: new hyperparameter configurations
            y_new: new performance metrics
        """
        if self.x_train is none:
            self.x_train = x_new
            self.y_train = y_new
        else:
            self.x_train = np.vstack((self.x_train, x_new))
            self.y_train = np.append(self.y_train, y_new)
        
        self.gp.fit(self.x_train, self.y_train)
