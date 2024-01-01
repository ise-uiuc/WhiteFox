
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 3)
  
  def forward(self, inputs):
    v1 = self.linear(inputs)
    v2 = v1 + other
    return v2

# Initializing the model
m = Model()

# Inputs to the model
inputs = torch.randn(1, 4, 4)
