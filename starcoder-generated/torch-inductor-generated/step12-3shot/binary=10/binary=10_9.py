
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(5, 10)
  
  def forward(self, x):
    x = self.linear(x)
    x = x + torch.nn.functional.pad(x, (0, 1))
  
    return x  # TODO

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5, 64, 64)
