"""
Integration tests for tinyHPO with MNIST.

These are smoke tests that verify the HPO methods work end-to-end
with a real model, using a reduced number of trials for speed.
"""

import sys
import os
import pytest

# Add src and examples to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from strategies import GridSearch, RandomSearch, BayesianSearch

# Skip if tinygrad not available
pytest.importorskip("tinygrad")

from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mnist_data():
    """Load MNIST data once for all tests."""
    X_train, Y_train, X_test, Y_test = mnist()
    # Use subset for faster tests
    return X_train[:1000], Y_train[:1000], X_test[:200], Y_test[:200]


@pytest.fixture
def param_grid():
    """HPO parameter grid for MNIST."""
    return {
        "hidden_size": [64, 128], # this wont be supported till later...
        "dropout": [0.0, 0.5],
        "learning_rate": [0.001, 0.01],
    }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class SimpleMNISTNet:
    """Simple MNIST classifier for testing."""

    def __init__(self, hidden_size: int = 128, dropout: float = 0.5) -> None:
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.fc1 = nn.Linear(2704, hidden_size)  # 16 * 13 * 13
        self.fc2 = nn.Linear(hidden_size, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu().max_pool2d((2, 2))
        x = x.flatten(1).dropout(self.dropout)
        x = self.fc1(x).relu()
        return self.fc2(x)


def quick_train(model, X_train, Y_train, learning_rate, steps=10):
    """Quick training for smoke tests."""
    optim = nn.optim.Adam(nn.state.get_parameters(model), lr=learning_rate)
    batch_size = 64

    for _ in range(steps):
        Tensor.training = True
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        x, y = X_train[samples], Y_train[samples]
        optim.zero_grad()
        loss = model(x).sparse_categorical_crossentropy(y)
        loss.backward()
        optim.step()


def evaluate(model, X_test, Y_test):
    """Evaluate model accuracy."""
    Tensor.training = False
    acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
    return acc


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

class TestMNISTIntegration:
    """Integration tests with MNIST model."""

    def test_grid_search_mnist(self, mnist_data, param_grid):
        """Test GridSearch with MNIST model."""
        X_train, Y_train, X_test, Y_test = mnist_data
        search = GridSearch(param_grid)

        results = []
        for trial in range(3):  # Only 3 trials for speed
            params = search()
            if params is None:
                break

            model = SimpleMNISTNet(
                hidden_size=params['hidden_size'],
                dropout=params['dropout']
            )
            quick_train(model, X_train, Y_train, params['learning_rate'])
            acc = evaluate(model, X_test, Y_test)

            results.append({'params': params, 'accuracy': acc})

        assert len(results) == 3
        assert all(0 <= r['accuracy'] <= 1 for r in results)
        assert all('hidden_size' in r['params'] for r in results)

    def test_random_search_mnist(self, mnist_data, param_grid):
        """Test RandomSearch with MNIST model."""
        X_train, Y_train, X_test, Y_test = mnist_data
        search = RandomSearch(param_grid, seed=42)

        results = []
        for trial in range(3):
            params = search()

            model = SimpleMNISTNet(
                hidden_size=params['hidden_size'],
                dropout=params['dropout']
            )
            quick_train(model, X_train, Y_train, params['learning_rate'])
            acc = evaluate(model, X_test, Y_test)

            results.append({'params': params, 'accuracy': acc})

        assert len(results) == 3
        assert all(0 <= r['accuracy'] <= 1 for r in results)

    def test_bayesian_search_mnist(self, mnist_data, param_grid):
        """Test BayesianSearch with MNIST model."""
        X_train, Y_train, X_test, Y_test = mnist_data
        search = BayesianSearch(param_grid, n_initial=2, seed=42)

        results = []
        for trial in range(4):
            params = search()
            if params is None:
                break

            model = SimpleMNISTNet(
                hidden_size=params['hidden_size'],
                dropout=params['dropout']
            )
            quick_train(model, X_train, Y_train, params['learning_rate'])
            acc = evaluate(model, X_test, Y_test)

            # Update Bayesian search with result
            search.update(params, acc)

            results.append({'params': params, 'accuracy': acc})

        assert len(results) >= 3
        assert all(0 <= r['accuracy'] <= 1 for r in results)

        # Check that Bayesian search tracked observations
        assert len(search.X_observed) == len(results)

    def test_model_instantiation_with_all_param_combinations(self, param_grid):
        """Verify model can be instantiated with all valid param combinations."""
        search = GridSearch(param_grid)

        for params in search.get_all():
            model = SimpleMNISTNet(
                hidden_size=params['hidden_size'],
                dropout=params['dropout']
            )
            assert model.hidden_size == params['hidden_size']
            assert model.dropout == params['dropout']

    def test_returned_params_within_bounds(self, param_grid):
        """Verify all methods return params within defined bounds."""
        strategies = [
            GridSearch(param_grid),
            RandomSearch(param_grid, seed=42),
            BayesianSearch(param_grid, n_initial=2, seed=42),
        ]

        for strategy in strategies:
            for _ in range(5):
                params = strategy()
                if params is None:
                    break

                # Update Bayesian if needed
                if hasattr(strategy, 'update'):
                    strategy.update(params, 0.5)

                # Check bounds
                assert params['hidden_size'] in param_grid['hidden_size']
                assert params['dropout'] in param_grid['dropout']
                assert params['learning_rate'] in param_grid['learning_rate']


class TestSearchFactory:
    """Test the search factory function."""

    def test_factory_creates_correct_strategies(self, param_grid):
        """Test get_hpo_strategy factory function."""
        from search import get_hpo_strategy

        grid = get_hpo_strategy('grid', param_grid)
        assert isinstance(grid, GridSearch)

        random = get_hpo_strategy('random', param_grid, seed=42)
        assert isinstance(random, RandomSearch)

        bayesian = get_hpo_strategy('bayesian', param_grid, seed=42)
        assert isinstance(bayesian, BayesianSearch)

    def test_factory_invalid_strategy(self, param_grid):
        """Test factory raises error for invalid strategy."""
        from search import get_hpo_strategy

        with pytest.raises(ValueError):
            get_hpo_strategy('invalid', param_grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
