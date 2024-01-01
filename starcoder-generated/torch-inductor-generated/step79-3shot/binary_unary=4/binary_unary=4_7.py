
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(2, 3, bias=False)
 
  def forward(self, x1, other=None):
    if other is None:
      other = torch.randn(2, 3)
    t1 = self.linear(x1)
    t2 = t1 + other
    t3 = relu(t2)
    return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
