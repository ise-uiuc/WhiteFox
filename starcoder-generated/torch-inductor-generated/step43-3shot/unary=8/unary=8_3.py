
class Model1(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_transpose1 = torch.nn.ConvTranspose2d(5, 9, 2, stride=1)
    self.conv_transpose2 = torch.nn.ConvTranspose2d(9, 12, 3, stride=1, padding=1, output_padding=0)
  def forward(self, x1):
    v1 = self.conv_transpose1(x1)
    v2 = self.conv_transpose2(v1)
    v3 = v2 + 3
    v4 = torch.clamp(v3, min=0)
    v5 = torch.clamp(v4, max=6)
    v6 = v1 * v5
    v7 = v6 / 6
    return v7

class Model2(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 11, 2, stride=1)
    self.conv_transpose2 = torch.nn.ConvTranspose2d(11, 13, 3, stride=1, padding=1, output_padding=0)
  def forward(self, x1):
    v1 = self.conv_transpose1(x1)
    v2 = self.conv_transpose2(v1)
    v3 = v2 + 3
    v4 = torch.clamp(v3, min=0)
    v5 = torch.clamp(v4, max=6)
    v6 = v1 * v5
    v7 = v6 / 6
    return v7

x1 = torch.randn(1, 5, 64, 64)
x2 = torch.rand(1, 3, 32, 32)
