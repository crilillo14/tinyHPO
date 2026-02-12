# tinyHPO - Project Guide

Minimal hyperparameter optimization library for tinygrad models.
Single entry point: `HyperparameterOptimizer`. Three search strategies: random, grid, bayesian.

## Architecture Decisions

### Optuna-style Interface (core design)

The library follows an **Optuna-style** architecture: the user provides a `SearchSpaceConfig` and an `objective(config: dict) -> float` callable. The library only does suggest/update. It never owns the model, optimizer, or data.

```python
config = SearchSpaceConfig() \
    .set_search("bayesian", acquisition_function="ei") \
    .set_iterations(50) \
    .append_hyperparameter("lr", (1e-5, 1e-1, 20), logscale=True) \
    .append_hyperparameter("dropout", (0.1, 0.5, 10))

def objective(params: dict) -> float:
    model = MyModel(dropout=params["dropout"])
    train(model, X_train, Y_train, lr=params["lr"])
    return evaluate(model, X_test, Y_test)

hpo = HyperparameterOptimizer(config, objective, maximize=True)
results = hpo.optimize()
```

The `HyperparameterOptimizer` orchestrates the suggest-evaluate-update loop. It handles logging, history tracking, early stopping, and visualization internally -- the user just provides config + callable.

**Rationale**: tinygrad models have no `nn.Module` base class, no `state_dict`, no standard parameter injection. Trying to own the training loop fights tinygrad's minimalism. The user's objective function is a black box that encapsulates all tinygrad-specific logic.

### numpy/scipy for GP Math (not tinygrad)

All surrogate model computation (GP, kernels, acquisition functions, Cholesky decomposition) uses **numpy and scipy**. Not tinygrad Tensors.

**Rationale**: GP matrices are tiny (~10-200 observed points, 2-10 dimensions). Cholesky on a 200x200 matrix is microseconds on CPU. The bottleneck is always the user's model evaluation, never the surrogate model. tinygrad Tensors would add GPU transfer overhead and complexity for zero gain.

tinygrad is relevant only at the interface boundary (examples, optional helpers). The library internals are numpy/scipy.

### tinyJIT: Not for library internals

tinyJIT accelerates repeated Tensor operations with fixed shapes. The GP's observation set grows each iteration (variable shapes), so tinyJIT doesn't apply. Users should `@TinyJIT` their own training functions. The library should document this, not enforce it.

### Candidate Sampling for Acquisition Optimization

`BayesianSearch._optimize_acquisition` uses **candidate sampling**: sample ~1000-5000 points from the parameter space, evaluate acquisition function on all (vectorized), pick argmax. No `scipy.optimize.minimize`.

**Rationale**: For <10 hyperparameters (typical HPO), candidate sampling is simple, fully vectorized, and effective. L-BFGS-B with restarts adds complexity for marginal gain in this regime.

### Minimize vs Maximize

The `SearchStrategy` interface always minimizes internally. When `maximize=True`, `HyperparameterOptimizer` negates the score before passing to `strategy.update()`. Strategies never know about the direction.

## File Structure

```
src/
  __init__.py          # Package exports
  hpo.py               # HyperparameterOptimizer (orchestrator, entry point)
  search.py            # SearchStrategy ABC + get_hpo_strategy factory
  types.py             # Hyperparameter dataclasses, ParameterSpace type alias
  strategies/
    __init__.py
    gridsearch.py      # GridSearch(SearchStrategy)
    randomsearch.py    # RandomSearch(SearchStrategy)
    bayesiansearch.py  # BayesianSearch(SearchStrategy), owns GP
    bayesian/
      __init__.py
      gaussianprocess.py  # GaussianProcess (surrogate model)
      kernels.py          # Kernel ABC + 8 implementations
      acquisition.py      # EI, PI, LCB + factory
  utils/
    wrap.py            # SearchSpaceConfig builder
    unpack.py          # Config key normalization (pseudonyms)
  viz/
    __init__.py
    plots.py           # matplotlib visualization suite
examples/
  mnist.py             # MNIST example with tinygrad
tests/
  test.py
  test_mnist.py        # Integration tests
  unit/
    gp.py              # GP unit tests (placeholder)
  models/
    mnist.py
```

## Code Conventions

### Style
- Loose type hints on public API signatures. No need for exhaustive annotation on internals.
- Minimal docstrings: one-liner for obvious methods, multi-line only for complex/non-obvious behavior.
- PEP 8 naming. Classes are CamelCase, functions/methods are snake_case.
- Dataclasses for value types (hyperparameters, configs).

### Imports
- **Within `src/`**: use relative imports. `from .types import ParameterSpace`, `from .strategies.bayesian.kernels import Kernel`.
- **From tests/examples**: use absolute imports. `from src.types import ParameterSpace`.
- Never mix relative and absolute within the same file.
- Group: stdlib, third-party, local. One blank line between groups.

### Patterns
- **ABC** for interfaces (`SearchStrategy`, `Kernel`, `Hyperparameter`).
- **Factory** for strategy selection (`get_hpo_strategy()`).
- **Builder** for config (`SearchSpaceConfig` with method chaining via `return self`).
- **Suggest-update loop** as the core optimization protocol:
  ```
  while not strategy.done:
      config = strategy.suggest()
      score = objective(config)
      strategy.update(config, score)
  ```

### What NOT to do
- Don't add tinygrad Tensor operations to GP/kernel/acquisition code.
- Don't create abstract base classes for things with only one implementation.
- Don't add optional dependencies without strong justification.
- Don't add compatibility shims or backwards-compat code. Just change it.

## Known Bugs (current state)

Priority order for fixing:

1. **`SearchSpaceConfig.append_hyperparameter`** (`wrap.py:73`): `space is tuple` should be `isinstance(space, tuple)`. Current code always raises ValueError.
2. **`SquaredExponential` kernel** (`kernels.py:31-33`): `X1 - X2` doesn't produce NxM pairwise distance matrix. Broken for GP use. Either fix or remove (it's a duplicate of RBF).
3. **`RandomSearch.suggest`** (`randomsearch.py:36`): Calls `h.sample()` but `Hyperparameter` has no `sample()` method. Should call `h(1)` or add `sample()`.
4. **`GridSearch._build_grid`** (`gridsearch.py:24`): Accesses `.values` attribute which doesn't exist on `Hyperparameter`. Need a `values` property that returns `np.linspace(lower, upper, partitions)`.
5. **Missing methods in `BayesianSearch`** (`bayesiansearch.py:58`): `_optimize_acquisition`, `_config_to_array`, `_array_to_config` referenced but not defined.
6. **Import mismatch in `hpo.py`** (line 6): Imports `GaussianProcessRegressor` but class is `GaussianProcess`. Also `from search import` should be relative.
7. **Empty method bodies** (`hpo.py:86-89`): `fit()` and `bayesianFit()` have no body.
8. **Duplicate GP ownership of acquisition** (`gaussianprocess.py:30`): GP stores `_acqfn` but `BayesianSearch` also stores `acqfn`. Acquisition belongs to `BayesianSearch`, not GP.
9. **`unpack.py` pseudonym logic** (line 19): Checks `k in acceptable_pseudonyms` (checks dict keys) but should check if `k` is in the value sets to resolve pseudonyms.

## Implementation Roadmap

### Phase 1: Fix Foundation

Fix all bugs above. Standardize imports to relative within `src/`. Consolidate `RBF` and `SquaredExponential` into one correct implementation. Add `sample()` method and `values` property to `ScalarHyperparameter`. Remove test/debug code from `gaussianprocess.py` (`testsampling()` function at bottom).

### Phase 2: Core Loop

Redesign `HyperparameterOptimizer` for Optuna-style interface:
- Constructor takes `(config: SearchSpaceConfig, objective: Callable, maximize: bool)`.
- `optimize()` method runs the suggest-evaluate-update loop.
- Built-in logging, history tracking, early stopping, and viz generation inside the loop.
- Remove all model/optimizer/data/loss args.
- Remove `HPO` ABC (unnecessary indirection for one class).
- Remove `bayesianFit()` -- one `optimize()` method dispatches to the right strategy.

Implement `BayesianSearch` private methods:
- `_config_to_array(config)`: dict -> np.ndarray (ordered by parameterspace keys).
- `_array_to_config(x)`: np.ndarray -> dict.
- `_optimize_acquisition()`: candidate sampling approach -- sample N candidates, predict with GP, evaluate acquisition, return argmax.

### Phase 3: Polish

- Built-in early stopping (e.g., no improvement for N trials).
- Built-in logging (print progress, best score so far).
- History tracking: list of `{trial, params, score}` dicts (matches viz module's expected format).
- Fix `SearchSpaceConfig` to support list-based discrete spaces for `CategoricalHyperparameter`.
- Update `examples/mnist.py` to use the new API.

### Phase 4: Extend

- CategoricalHyperparameter support in search strategies (one-hot encoding for GP, direct sampling for random/grid).
- k-fold cross validation helper (optional utility, not core).
- Visualization integration into optimize loop (auto-generate plots on completion).
- PyPI packaging, wheel, CLI.

## Testing Strategy

- **Unit tests**: Each kernel produces correct-shape output. GP fit/predict returns correct shapes. Acquisition functions return correct shapes and respect monotonicity properties.
- **Integration tests**: Full optimize loop with a trivial objective (e.g., quadratic function) for each strategy. Verify convergence, bounds respect, history tracking.
- **MNIST smoke test**: One short run with each strategy on the real MNIST model. Not for correctness, just to verify nothing crashes end-to-end.
- Run with `pytest tests/` from project root.

## Dependencies

**Core** (always needed): `numpy`, `scipy`
**Visualization** (optional): `matplotlib`, `seaborn`
**User's domain**: `tinygrad` (used in examples/tests, not imported by library internals except at the interface layer)
**Testing**: `pytest`
