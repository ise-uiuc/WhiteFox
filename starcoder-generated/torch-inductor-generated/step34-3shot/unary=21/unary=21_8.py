
class ModelTanh(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 1, 11, stride=7, padding=5, dilation=2)

  def forward(self, x):
    t0 = self.conv1(x)
    t1 = torch.tanh(t0)
    return t1
# Inputs to the model
x=torch.randn(1, 3, 226, 226)
