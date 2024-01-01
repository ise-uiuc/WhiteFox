
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv2d = nn.Conv2d(3, 3, 1, 1, dilation=2, padding=4)
    self.conv2d1 = nn.Conv2d(3, 3, 1, 1)
    self.conv2d2 = nn.Conv2d(3, 3, 1, 1, dilation=2)
    self.conv2d3 = nn.Conv2d(3, 3, 1, 1, padding=5)
    self.conv2d4 = nn.Conv2d(3, 3, 1, 1)
  def forward(self, x):
    t1 = self.conv2d(x)
    t2 = self.conv2d1(x)
    t3 = self.conv2d2(x)
    t4 = self.conv2d3(x)
    t5 = self.conv2d4(x)
    t6 = t1 * t2 - t3 * t4 + t5
    return t6
# Inputs to the model
x = torch.randn(1, 3, 128, 64)
