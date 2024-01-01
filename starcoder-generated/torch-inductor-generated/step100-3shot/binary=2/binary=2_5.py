
class Model(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = torch.nn.Conv2d(4, 6, 4, stride=4, padding=4)
      self.conv2 = torch.nn.Conv2d(6, 2, 2, stride=2, padding=2)

    def forward(self, x):
      v1 = self.conv1(x)
      v2 = self.conv2(v1)
      v3 = v2 - 1
      return v3
# Inputs to the model
x = torch.rand(1, 4, 32, 32)
