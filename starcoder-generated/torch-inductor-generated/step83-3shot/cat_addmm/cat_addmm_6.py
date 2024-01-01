
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Linear(2, 2)
    self.layers_2 = nn.Linear(2, 2)
  def forward(self, x):
    x = self.layers_2(x)
    x = self.layers(x)
    x = torch.flatten(x, start_dim=1)
    x = x[0]
    return x
# Inputs to the model
x = torch.randn(2, 2)
