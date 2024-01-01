
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(1, 8, 5, stride=1, padding=2)
    self.conv2 = torch.nn.Conv2d(1, 8, 5, stride=1, padding=2)
  def forward(self, x1):
    t1 = self.conv1(x1)
    t2 = self.conv1(t1)
    t3 = self.conv1(t2)
    t4 = self.conv2(x1)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
