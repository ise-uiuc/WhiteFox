
class ClampModel(torch.nn.Module):
  def __init__(self, min=-0.75, max=-0.05):
    super().__init__()
    self.conv = torch.nn.Conv2d(1, 1, 3)
    self.min = min
    self.max = max
  def forward(self, x1):
    conv = self.conv(x1 - self.min)
    return torch.clamp(conv, min=-self.max)
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
