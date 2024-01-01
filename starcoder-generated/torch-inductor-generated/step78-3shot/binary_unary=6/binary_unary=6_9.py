
class Model(torch.nn.Module):
  def __init__(self, linear_transform):
      super().__init__()
      self.linear_transform = linear_transform

  def forward(self, x):
      v1 = self.linear_transform(x)
      v2 = v1 - 100
      v3 = torch.relu(v2)
      return v3

# Initializing the linear transformation
m = torch.nn.Linear(3, 4)

# Initializing the model
n = Model(m)

# Inputs to the model
x = torch.randn(1, 3)
