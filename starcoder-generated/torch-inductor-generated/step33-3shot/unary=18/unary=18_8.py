
class Model(torch.nn.Module):
    def __init__(self):
      super(Model, self).__init__()
      self.conv1 = torch.nn.Conv2d(1, 2, 1, stride=1, padding=1)
      self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=1, padding=1)
      self.conv3 = torch.nn.Conv2d(16, 8, 5, stride=1, padding=1)
    def forward(self, x1):
      v1 = self.conv1(x1)
      v2 = torch.sigmoid(v1)
      v3 = self.conv2(v2)
      v4 = torch.sigmoid(v3)
      v5 = self.conv3(v4)
      v6 = torch.sigmoid(v5)
      return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
