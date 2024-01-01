
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(20, 50)
 
  def forward(self, x1, other):
    v1 = self.linear(x1)
    v2 = v1 + other
    v3 = torch.relu(v2)
    return v3

# Initializing the model
batch = 20
n = 50
m = Model()

# Inputs to the model
x1 = torch.randn(batch, 20)
x2 = torch.randn(batch, n)
