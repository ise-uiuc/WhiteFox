
class Model(torch.nn.Module):
  def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(6, 8)
      self.linear1 = torch.nn.Linear(6, 16)

  def forward(self, x):
      v = self.linear(x)
      v1 = self.linear1(x)
      v2 = v + v1
      v3 = torch.nn.functional.relu(v2)
      return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 6)
