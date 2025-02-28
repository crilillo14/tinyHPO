from tinygrad.nn.datasets import mnist

# Load your data
X_train, Y_train, X_test, Y_test = mnist()

# Define your model class (must accept hyperparameters in __init__)
class MyModel:
    def __init__(self, hidden_size=128, dropout=0.0):
        # Model definition here
        pass
        
    def __call__(self, x):
        # Forward pass here
        return output

# Run optimization
best_params = optimize_model(
    model_class=MyModel,
    train_data=(X_train, Y_train),
    test_data=(X_test, Y_test),
    n_trials=20
)