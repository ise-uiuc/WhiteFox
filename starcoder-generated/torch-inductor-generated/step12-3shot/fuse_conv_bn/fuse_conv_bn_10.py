
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = torch.nn.Conv2d(3, 3, 3)
    self.bn = torch.nn.BatchNorm3d(3)
  def forward(self, x):
    x = self.conv(x)
    return self.bn(x)
# Inputs to the model
x = torch.randn(1, 3, 4, 4, 4)
