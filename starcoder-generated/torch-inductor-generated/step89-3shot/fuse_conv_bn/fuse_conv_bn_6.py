


class Model(nn.Module):
  def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(32, 32, (2, 11), stride=(2, 1))
      self.bn = nn.BatchNorm2d(32)

  def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      return x
# Inputs to the model
x = torch.randn(4, 32, 14, 31) 
