 2
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.relu6 = torch.nn.ReLU6()

  def forward(self, x1, x2):
    v1 = torch.cat([x1, x2], dim=1)
    v2 = v1[:, 0:9223372036854775807]
    v3 = v2[:, 0:6148914691236517205]
    v4 = torch.cat([v1, v3], dim = 1)
    v5 = self.relu6(v4)
    return v5

# Initializing the models
m1 = Model()
m2 = Model()

# Inputs to the models
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
m1(x1, x2)
