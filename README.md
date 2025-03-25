# TinyHPO (Hyper Paremeter Optimization)

Minimal hyperparameter optimization tooling compatible with tinygrad models. 

## Installing


### Via pip
Coming soon! Git clone for the time being, it's really not a big library.

### Via git clone

```bash
git clone https://github.com/crilillo14/tinyHPO.git
cd tinyHPO
```

## How to use

*Define a parameter space, and tinyHPO will search it for you!*

You can choose one of three methods for the search:
+ Grid Search (brute force search of the entire parameter space)
+ [Random](https://www.kaggle.com/code/willkoehrsen/intro-to-model-tuning-grid-and-random-search) (picks n random model hyperparameter configs to test)
+ Bayesian Search (...)

For small optimization jobs, grid search is best. 
Random Search is, you know, random. So gamble if you want, you might get an optimal solution in very few tries!
Bayesian Search is **best**. Less random, more of an educated guess. 

> Note : In the tinygrad docs, models are often, if not always defined as follows:

```python
class MNIST:
  def __init__(self):
      self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
      self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
      self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
      x = self.l1(x).relu().max_pool2d((2,2))
      x = self.l2(x).relu().max_pool2d((2,2))
      return self.l3(x.flatten(1).dropout(0.5))
```

This structure is minimalistic, lightweight, understandable -- but also hardcoded and tons more complicated to inspect and ultimately reconstruct when surveying the _parameter space_.

For the parameters you want to optimize, these should be defined as args in the constructor.

*Ex: Optimizing the*


### Study!

Inspired by [optuna's](https://optuna.org/#code_examples) optimization practices, tinyHPO revolves around one essential function; **study**.

```python
import tinyHPO as hpo

study = hpo.create_study()


```


