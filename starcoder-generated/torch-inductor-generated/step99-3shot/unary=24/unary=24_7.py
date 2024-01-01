
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(128, 64, 1, 1, 0)

  def forward(self, x):
    negative_slope = 0.6850005
    v1 = self.conv1(x)
    v2 = v1 > 0
    v3 = v1 * negative_slope
    v4 = torch.where(v2, v1, v3)
    return v4

# Inputs to the model
x1 = torch.randn(3, 128, 128, 16)
