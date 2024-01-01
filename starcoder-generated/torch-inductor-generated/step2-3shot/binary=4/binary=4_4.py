
class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(100, 300)
 
  def forward(self, x):
    v1 = self.linear(x)
    v2 = v1 + x
    return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(10, 100, dtype=torch.float32, requires_grad=True)
