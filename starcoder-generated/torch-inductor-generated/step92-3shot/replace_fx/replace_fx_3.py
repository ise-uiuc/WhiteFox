
class model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 2, 3)
    self.conv2 = nn.Conv2d(1, 2, 3)
  def forward(self, input):
    x = self.conv1(input)
    x = F.max_pool2d(x, 3, stride=2)
    x = self.conv2(x)
    return x
# Inputs to the model
input = torch.randn(16, 1, 5, 5)
