
class Model1(torch.nn.Module):
  def __init__(self):
    super(Model1, self).__init__()
    self.layers = []
    for i in range(3):
      layer = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2)
      self.layers.append(layer)
    self.layers = torch.nn.ModuleList(self.layers)
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
      x = torch.tanh(x)
    return x
# Inputs to the model
torch.manual_seed(0)
x = torch.randn(1, 3, 224, 224)
