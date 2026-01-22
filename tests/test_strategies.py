"""
Unit tests for tinyHPO search strategies.
"""

import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from strategies import GridSearch, RandomSearch, BayesianSearch


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_param_grid():
    """Simple 2x2 parameter grid for testing."""
    return {
        "a": [1, 2],
        "b": [10, 20],
    }


@pytest.fixture
def hpo_param_grid():
    """HPO-style parameter grid."""
    return {
        "hidden_size": [64, 128, 256],
        "dropout": [0.0, 0.25, 0.5],
        "learning_rate": [0.001, 0.01, 0.1],
    }


# -----------------------------------------------------------------------------
# GridSearch Tests
# -----------------------------------------------------------------------------

class TestGridSearch:
    """Tests for GridSearch."""

    def test_init(self, simple_param_grid):
        """Test GridSearch initialization."""
        search = GridSearch(simple_param_grid)
        assert search.num_elements == 4
        assert search.degrees_of_freedom == 2
        assert search.iteration == 0

    def test_exhaustive_iteration(self, simple_param_grid):
        """Test that GridSearch visits all combinations exactly once."""
        search = GridSearch(simple_param_grid)
        visited = []

        for _ in range(10):  # More than total combinations
            params = search()
            if params is None:
                break
            visited.append(tuple(sorted(params.items())))

        # Should visit exactly 4 combinations
        assert len(visited) == 4

        # All combinations should be unique
        assert len(set(visited)) == 4

        # Should contain all expected combinations
        expected = {
            (('a', 1), ('b', 10)),
            (('a', 2), ('b', 10)),
            (('a', 1), ('b', 20)),
            (('a', 2), ('b', 20)),
        }
        assert set(visited) == expected

    def test_get_all(self, simple_param_grid):
        """Test get_all returns all combinations."""
        search = GridSearch(simple_param_grid)
        all_params = search.get_all()

        assert len(all_params) == 4
        assert all(isinstance(p, dict) for p in all_params)

    def test_reset(self, simple_param_grid):
        """Test reset functionality."""
        search = GridSearch(simple_param_grid)

        # Consume some iterations
        search()
        search()

        assert search.iteration == 2

        # Reset
        search.reset()
        assert search.iteration == 0
        assert len(search.history) == 0

    def test_iterator_protocol(self, simple_param_grid):
        """Test that GridSearch supports iteration."""
        search = GridSearch(simple_param_grid)
        visited = list(search)

        assert len(visited) == 4

    def test_len(self, simple_param_grid):
        """Test __len__ returns total combinations."""
        search = GridSearch(simple_param_grid)
        assert len(search) == 4

    def test_larger_grid(self, hpo_param_grid):
        """Test with larger HPO-style grid."""
        search = GridSearch(hpo_param_grid)
        assert len(search) == 27  # 3 * 3 * 3


# -----------------------------------------------------------------------------
# RandomSearch Tests
# -----------------------------------------------------------------------------

class TestRandomSearch:
    """Tests for RandomSearch."""

    def test_init(self, simple_param_grid):
        """Test RandomSearch initialization."""
        search = RandomSearch(simple_param_grid, seed=42)
        assert search.seed == 42
        assert search.iteration == 0

    def test_produces_valid_params(self, simple_param_grid):
        """Test that RandomSearch produces valid parameter combinations."""
        search = RandomSearch(simple_param_grid, seed=42)

        for _ in range(10):
            params = search()
            assert 'a' in params
            assert 'b' in params
            assert params['a'] in [1, 2]
            assert params['b'] in [10, 20]

    def test_seed_reproducibility(self, simple_param_grid):
        """Test that same seed produces same sequence."""
        search1 = RandomSearch(simple_param_grid, seed=42)
        search2 = RandomSearch(simple_param_grid, seed=42)

        for _ in range(5):
            p1 = search1()
            p2 = search2()
            assert p1 == p2

    def test_different_seeds_different_results(self, simple_param_grid):
        """Test that different seeds produce different sequences."""
        search1 = RandomSearch(simple_param_grid, seed=42)
        search2 = RandomSearch(simple_param_grid, seed=123)

        # Sample multiple times and compare
        results1 = [search1() for _ in range(10)]
        results2 = [search2() for _ in range(10)]

        # Should be different (with high probability)
        assert results1 != results2

    def test_sample_method(self, simple_param_grid):
        """Test the sample() method."""
        search = RandomSearch(simple_param_grid, seed=42)
        samples = search.sample(5)

        assert len(samples) == 5
        assert all(isinstance(s, dict) for s in samples)

    def test_reset(self, simple_param_grid):
        """Test reset with new seed."""
        search = RandomSearch(simple_param_grid, seed=42)

        # Sample some
        first_sample = search()

        # Reset with same seed
        search.reset(seed=42)
        after_reset = search()

        assert first_sample == after_reset

    def test_history_tracking(self, simple_param_grid):
        """Test that history is tracked."""
        search = RandomSearch(simple_param_grid, seed=42)

        for _ in range(5):
            search()

        assert len(search.history) == 5


# -----------------------------------------------------------------------------
# BayesianSearch Tests
# -----------------------------------------------------------------------------

class TestBayesianSearch:
    """Tests for BayesianSearch."""

    def test_init(self, simple_param_grid):
        """Test BayesianSearch initialization."""
        search = BayesianSearch(simple_param_grid, seed=42)
        assert search.iteration == 0
        assert len(search.X_observed) == 0

    def test_produces_valid_params(self, simple_param_grid):
        """Test that BayesianSearch produces valid parameter combinations."""
        search = BayesianSearch(simple_param_grid, n_initial=2, seed=42)

        for _ in range(4):
            params = search()
            if params is None:
                break
            assert 'a' in params
            assert 'b' in params
            assert params['a'] in [1, 2]
            assert params['b'] in [10, 20]

    def test_update_and_suggest(self, simple_param_grid):
        """Test update() and suggestion flow."""
        search = BayesianSearch(simple_param_grid, n_initial=2, seed=42)

        # Initial random samples
        for i in range(2):
            params = search()
            score = float(params['a'] + params['b'])  # Simple objective
            search.update(params, score)

        assert len(search.X_observed) == 2
        assert len(search.y_observed) == 2

        # Now should use GP for suggestions
        params = search()
        assert params is not None

    def test_get_best(self, simple_param_grid):
        """Test get_best() method."""
        search = BayesianSearch(simple_param_grid, n_initial=2, seed=42)

        # Run a few iterations
        for i in range(3):
            params = search()
            score = float(params['a'] + params['b'])
            search.update(params, score)

        best_params, best_score = search.get_best()
        assert best_params is not None
        assert best_score is not None

        # Best should have highest score
        assert best_score == max(search.y_observed)

    def test_exhaustion(self, simple_param_grid):
        """Test that search returns None when exhausted."""
        search = BayesianSearch(simple_param_grid, n_initial=2, seed=42)

        # Try more than total combinations
        for i in range(10):
            params = search()
            if params is None:
                break
            search.update(params, float(i))

        # Should have tried all 4 combinations
        assert len(search.X_observed) == 4

    def test_different_acquisition_functions(self, simple_param_grid):
        """Test different acquisition functions can be used."""
        for acq in ['ei', 'ucb', 'pi']:
            search = BayesianSearch(simple_param_grid, acquisition=acq, seed=42)
            assert search.acquisition_name == acq

            # Should work without errors
            for _ in range(3):
                params = search()
                if params:
                    search.update(params, 1.0)

    def test_reset(self, simple_param_grid):
        """Test reset functionality."""
        search = BayesianSearch(simple_param_grid, n_initial=2, seed=42)

        # Run some iterations
        for _ in range(3):
            params = search()
            if params:
                search.update(params, 1.0)

        # Reset
        search.reset()

        assert len(search.X_observed) == 0
        assert len(search.y_observed) == 0
        assert search.iteration == 0


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for all strategies."""

    def test_all_strategies_same_grid(self, hpo_param_grid):
        """Test all strategies can work with the same parameter grid."""
        grid = GridSearch(hpo_param_grid)
        random = RandomSearch(hpo_param_grid, seed=42)
        bayesian = BayesianSearch(hpo_param_grid, n_initial=3, seed=42)

        # All should produce valid params
        g_params = grid()
        r_params = random()
        b_params = bayesian()

        for params in [g_params, r_params, b_params]:
            assert set(params.keys()) == {'hidden_size', 'dropout', 'learning_rate'}
            assert params['hidden_size'] in [64, 128, 256]
            assert params['dropout'] in [0.0, 0.25, 0.5]
            assert params['learning_rate'] in [0.001, 0.01, 0.1]

    def test_mock_optimization_loop(self, simple_param_grid):
        """Test a mock optimization loop with each strategy."""

        def mock_objective(params):
            """Higher is better when a=2 and b=20."""
            return params['a'] * 0.5 + params['b'] * 0.01

        for Strategy, kwargs in [
            (GridSearch, {}),
            (RandomSearch, {'seed': 42}),
            (BayesianSearch, {'n_initial': 2, 'seed': 42}),
        ]:
            search = Strategy(simple_param_grid, **kwargs)
            best_score = float('-inf')
            best_params = None

            for _ in range(4):
                params = search()
                if params is None:
                    break

                score = mock_objective(params)

                if hasattr(search, 'update'):
                    search.update(params, score)

                if score > best_score:
                    best_score = score
                    best_params = params

            # Should find a good solution
            assert best_params is not None
            assert best_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
