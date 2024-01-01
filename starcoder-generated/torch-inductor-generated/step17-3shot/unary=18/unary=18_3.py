
class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
  def forward(self, x):
    v1 = self.conv1(x)
    return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
