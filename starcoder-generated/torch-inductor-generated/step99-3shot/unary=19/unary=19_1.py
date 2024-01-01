
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(100, 1)

  def forward(self, x1):
    o1 = self.linear(x1)
    y = torch.sigmoid(o1)
    return y

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
y = model(x1)

