# tinygrad intro in docs ... 
# note : should change the structure of the class such that the values 


from tinygrad import device, tensor, nn
from tinygrad.nn.datasets import mnist

# outputs metal :: apple silicon accelerator framework
print(device.default)

class mnist: 
    def __init__(self) -> none:
      self.l1 = nn.conv2d(1, 32, kernel_size=(3,3))
      self.l2 = nn.conv2d(32, 64, kernel_size=(3,3))
      self.l3 = nn.linear(1600, 10)


class mnist:
  def __init__(self):
      self.l1 = nn.conv2d(1, 32, kernel_size=(3,3))
      self.l2 = nn.conv2d(32, 64, kernel_size=(3,3))
      self.l3 = nn.linear(1600, 10)

  def __call__(self, x:tensor) -> tensor:
      x = self.l1(x).relu().max_pool2d((2,2))
      x = self.l2(x).relu().max_pool2d((2,2))
      return self.l3(x.flatten(1).dropout(0.5))

# -----------------------------------------------------------------------

x_train, y_train, x_test, y_test = mnist()
print(x_train.shape, x_train.dtype, y_train.shape, y_train.dtype)
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar


# -----------------------------------------------------------------------


model = mnist()
acc = (model(x_test).argmax(axis=1) == y_test).mean()
# note: tinygrad is lazy, and hasn't actually run anything by this point
print(acc.item())  # ~10% accuracy, as expected from a random model


# -----------------------------------------------------------------------


optim = nn.optim.adam(nn.state.get_parameters(model))
batch_size = 128
def step():
  tensor.training = true  # makes dropout work 
  samples = tensor.randint(batch_size, high=x_train.shape[0])
  x, y = x_train[samples], y_train[samples]
  optim.zero_grad()
  loss = model(x).sparse_categorical_crossentropy(y).backward()
  optim.step()
  return loss


# -----------------------------------------------------------------------


import timeit
timeit.repeat(step, repeat=5, number=1)
#[0.08268719699981375,
# 0.07478952900009972,
# 0.07714716600003158,
# 0.07785399599970333,
# 0.07605237000007037]

# -----------------------------------------------------------------------

from tinygrad import tiny_jit
jit_step = tiny_jit(step)

timeit.repeat(jit_step, repeat=5, number=1)
# [0.2596786549997887,
#  0.08989566299987928,
#  0.0012115650001760514,
#  0.001010227999813651,
#  0.0012164899999334011]


