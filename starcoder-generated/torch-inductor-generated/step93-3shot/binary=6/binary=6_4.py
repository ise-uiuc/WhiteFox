
class Model(torch.nn.Module):
  def __init__(self, outputSize)
      super().__init__()
      self.linear1 = torch.nn.Linear(outputSize, outputSize // 2)
 
  def forward(self, x):
      v1 = self.linear1(x)
      v2 = v1 - 1
      return v1, v2

# Initializing the model
outputSize = 10
m = Model(outputSize)

# Inputs to the model
x = torch.tensor([[1, 2, 3]])
x1 = torch.randn(2, 3, 64, 64)
__output__, __ouput__ = m(x)

x1 = torch.randn(2, 3, 64, 64)
__output__, __output2__ = m(x1)

