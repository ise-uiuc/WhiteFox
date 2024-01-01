
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(5, 3)

  def forward(self, x, other=None):
    v1 = self.linear(x)
    v2 = v1 + other
    v3 = v2.relu()
    return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
