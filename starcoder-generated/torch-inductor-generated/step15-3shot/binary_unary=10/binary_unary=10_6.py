
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(3, 8, bias=False)

  def forward(self, x1, x2):
   v1 = self.linear(x1)
   v2 = v1 + x2
   v3 = v2.relu()
   return v3

# Initializing the model
m = Model()
m.eval()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 8)
__output = m(x1, x2)

