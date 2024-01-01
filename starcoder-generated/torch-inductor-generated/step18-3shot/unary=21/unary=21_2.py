

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(4, 13, 1, stride=1, padding=0)
    self.t1 = torch.nn.Tanh()
    self.conv2 = torch.nn.Conv2d(13, 3072, 11, stride=11, padding=0)
    self.t2 = torch.nn.Tanh()
    self.conv3 = torch.nn.Conv2d(3072, 1000, 1, stride=1, padding=0)
    self.t3 = torch.nn.Tanh()
  def forward(self, x):
    v1 = self.conv1(x)
    v2 = self.t1(v1)
    v3 = self.conv2(v2)
    v4 = self.t2(v3)
    v5 = self.conv3(v4)
    v6 = self.t3(v5)
    return v6

# Inputs to the model
x = torch.randn(1,4,318,255)
