from optimizer import hyperparam_optim
from tinygrad.nn.datasets import mnist

# load your data
x_train, y_train, x_test, y_test = mnist()

# define your model class (must accept hyperparameters in __init__)
class my_model:
    def __init__(self, hidden_size=128, dropout=0.0):
        # model definition here
        pass
        
    def __call__(self, x):
        # forward pass here
        return output

# run optimization
best_params = hyperparam_optim.optimize_model(
    model_class=my_model,
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    n_trials=20
)