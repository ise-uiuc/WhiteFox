
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1)
    self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
  def forward(self, x):
    v1 = self.conv1(x)
    v2 = 1 + v1
    v3 = self.conv2(v2)
    v4 = v3 + x
    v5 = torch.relu(v4)
    v6 = v4 - x
    v7 = torch.relu(v6)
    return v7
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
