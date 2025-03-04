from tinygrad import Tensor, nn
from tinygrad.examples.mnist import fetch_mnist
import numpy as np
from hyperparamOptim import HyperparameterOptimizer, OptimizationResult

class SimpleMNISTNet:
    def __init__(self, hidden_size: int = 128, learning_rate: float = 0.01):
        self.l1 = nn.Linear(784, int(hidden_size))  # 28*28 = 784
        self.l2 = nn.Linear(int(hidden_size), 10)
        self.learning_rate = learning_rate
        
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).log_softmax()
        return x
    
    def train(self, x, y, batch_size=128, epochs=3):
        x = x.reshape(-1, 784)  # Flatten 28x28 images
        
        # Convert labels to one-hot
        y_onehot = np.zeros((y.shape[0], 10))
        y_onehot[range(y.shape[0]), y] = 1
        y = Tensor(y_onehot)
        
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                out = self(batch_x)
                loss = -(out * batch_y).sum() / batch_size
                
                # Backward pass
                loss.backward()
                
                # Update weights
                for layer in [self.l1, self.l2]:
                    layer.weight = layer.weight - layer.weight.grad * self.learning_rate
                    layer.bias = layer.bias - layer.bias.grad * self.learning_rate
                    
                    # Zero gradients
                    layer.weight.grad = None
                    layer.bias.grad = None

def accuracy_metric(y_true, y_pred):
    """Calculate classification accuracy"""
    pred_class = y_pred.numpy().argmax(axis=1)
    true_class = y_true.numpy().argmax(axis=1)
    return -float((pred_class == true_class).mean())  # Negative because we want to maximize accuracy

def main():
    # Load MNIST data using tinygrad's fetch_mnist
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    
    # Convert to tensors and normalize
    X_train, Y_train = Tensor(X_train).float()/255.0, Tensor(Y_train)
    X_test, Y_test = Tensor(X_test).float()/255.0, Tensor(Y_test)
    
    # Define hyperparameter space
    hyperparam_space = {
        'hidden_size': (32, 256),  # Will be converted to int in model
        'learning_rate': (0.0001, 0.1)
    }
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        model_class=SimpleMNISTNet,
        train_data=(X_train, Y_train),
        hyperparam_space=hyperparam_space,
        metric=accuracy_metric,
        minimize=True  # We're using negative accuracy, so we minimize
    )
    
    # Run optimization
    print("Starting Bayesian optimization...")
    results = optimizer.bayesian_optimize(n_iterations=20)
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best parameters: {results.best_params}")
    print(f"Best accuracy: {-results.best_score:.4f}")  # Convert back to positive accuracy
    
    # Train final model with best parameters
    best_model = SimpleMNISTNet(**results.best_params)
    best_model.train(X_train, Y_train.numpy())
    
    # Evaluate on test set
    X_test = X_test.reshape(-1, 784)  # Flatten test images
    with Tensor.no_grad():
        test_pred = best_model(X_test)
        test_accuracy = -accuracy_metric(
            Tensor(np.eye(10)[Y_test.numpy()]), 
            test_pred
        )
    print(f"\nTest accuracy with best model: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
