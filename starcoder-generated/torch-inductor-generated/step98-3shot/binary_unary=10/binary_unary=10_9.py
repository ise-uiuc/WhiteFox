
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(6, 8, False)

  def forward(self, x1, other):
    t1 = self.linear(x1)
    t2 = t1 + other
    t3 = torch.sigmoid(t2)
    return t3

# Initialization of the model
m = Model()

# Inputs to the model
x1 = torch.randn(28, 6)

