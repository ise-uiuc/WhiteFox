
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(1, 16, 1, padding=(1, 4))
    self.conv2 = torch.nn.Conv2d(16, 8, 1, padding=(2, 0))
    self.conv3 = torch.nn.Conv2d(8, 24, 1, padding=(0, 1))
  def forward(self, x1, x2, x3, w1):
    var1 = self.conv1(x1)
    var2 = self.conv2(x2)
    var3 = self.conv3(x3)
    var4 = var1 + var2
    var5 = var1 - var2
    var6 = var4 + var3
    var7 = var5 + var3 + w1
    var8 = var5 * w1
    return [var6, var7, var8]
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
x2 = torch.randn(1, 1, 128, 128)
x3 = torch.randn(1, 1, 128, 128)
w1 = torch.randn(1, 1)
