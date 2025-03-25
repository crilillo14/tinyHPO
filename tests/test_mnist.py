from tinygrad import tensor, nn
from tinygrad.examples.mnist import fetch_mnist
import numpy as np
from tinyhpo import hyperparameter_optimizer, optimization_result

class simple_mnistnet:
    def __init__(self, hidden_size: int = 128, learning_rate: float = 0.01):
        self.l1 = nn.linear(784, int(hidden_size))  # 28*28 = 784
        self.l2 = nn.linear(int(hidden_size), 10)
        self.learning_rate = learning_rate
        
    def __call__(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).log_softmax()
        return x
    
    def train(self, x, y, batch_size=128, epochs=3):
        x = x.reshape(-1, 784)  # flatten 28x28 images
        
        # convert labels to one-hot
        y_onehot = np.zeros((y.shape[0], 10))
        y_onehot[range(y.shape[0]), y] = 1
        y = tensor(y_onehot)
        
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # forward pass
                out = self(batch_x)
                loss = -(out * batch_y).sum() / batch_size
                
                # backward pass
                loss.backward()
                
                # update weights
                for layer in [self.l1, self.l2]:
                    layer.weight = layer.weight - layer.weight.grad * self.learning_rate
                    layer.bias = layer.bias - layer.bias.grad * self.learning_rate
                    
                    # zero gradients
                    layer.weight.grad = none
                    layer.bias.grad = none

def accuracy_metric(y_true, y_pred):
    """calculate classification accuracy"""
    pred_class = y_pred.numpy().argmax(axis=1)
    true_class = y_true.numpy().argmax(axis=1)
    return -float((pred_class == true_class).mean())  # negative because we want to maximize accuracy

def main():
    # load mnist data using tinygrad's fetch_mnist
    x_train, y_train, x_test, y_test = fetch_mnist()
    
    # convert to tensors and normalize
    x_train, y_train = tensor(x_train).float()/255.0, tensor(y_train)
    x_test, y_test = tensor(x_test).float()/255.0, tensor(y_test)
    
    # define hyperparameter space
    hyperparam_space = {
        'hidden_size': (32, 256),  # will be converted to int in model
        'learning_rate': (0.0001, 0.1)
    }
    
    # create optimizer
    optimizer = hyperparameter_optimizer(
        model_class=simple_mnistnet,
        train_data=(x_train, y_train),
        hyperparam_space=hyperparam_space,
        metric=accuracy_metric,
        minimize=true  # we're using negative accuracy, so we minimize
    )
    
    # run optimization
    print("starting bayesian optimization...")
    results = optimizer.bayesian_optimize(n_iterations=20)
    
    # print results
    print("\n_optimization results:")
    print(f"best parameters: {results.best_params}")
    print(f"best accuracy: {-results.best_score:.4f}")  # convert back to positive accuracy
    
    # train final model with best parameters
    best_model = simple_mnistnet(**results.best_params)
    best_model.train(x_train, y_train.numpy())
    
    # evaluate on test set
    x_test = x_test.reshape(-1, 784)  # flatten test images
    with tensor.no_grad():
        test_pred = best_model(x_test)
        test_accuracy = -accuracy_metric(
            tensor(np.eye(10)[y_test.numpy()]), 
            test_pred
        )
    print(f"\n_test accuracy with best model: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
