
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(3, 5)

  def forward(self, x):
    v1 = self.linear(x)
    v2 = v1 > 0
    v3 = v1 * 0.01
    v4 = torch.where(v2, v1, v3)
    return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3)
