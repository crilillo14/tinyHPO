

You have one side of this being interacting with the model, optimizer and data to evaluate the costly function f(X).
Because of this, ´´´src.hpo´´´ and  ´´´src.search´´´ needs to deal with tinygrad Tensors.

under src/strategy, np, sp, and sk learn can be assumed to be used.

Have to consider whether cross validation will also be used.

> f(x) cannot be simply measured as the accuracy on one sample, as f(x) is really the *global* loss, which is intractable. it can be approximated with F(X) which is either the mean of a full train test sample, the model being trained on trainX and measured against testX, or the cross-validated mean of the model on the full dset X.
--- 

### Design:

These are more or less the steps of this lib's workflow.
1. Canonicalize model, optim and dset in a single Model type. Could be hpo.pack(model, optim, dset, ..., strategy)
    1. have to consider whether to include nn.state
    2. config dclass? Maybe
2. *backend* construct strategy types and acq fn (if bayesian). 
3. iterate over exploration. if trained once, user should give the hp states and loss val.
    1. for each tensor H that we predict, fit and eval with the given callables by the user. 
4. return hp_state_dict, save viz, save state hist.



The imagined flow of this is:

´´´python

import tinygrad, tinyhpo

class UserModel:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))

optim = tinygrad.optim.(*)
X_train, Y_train, X_test, Y_test = (*)
lossFunc = ( * )

config = {
model 
}

hpo = tinyhpo()

hpo.optimize(n_steps=10, save_viz=True, save_trials=False, output_dir : str = 'optimization')
´´´
