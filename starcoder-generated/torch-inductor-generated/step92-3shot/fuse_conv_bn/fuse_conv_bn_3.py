
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x, training):
    x1 = self.conv(x)
    y = self.bn(x1, training)
    return y
  
  def conv(self, x):
    conv = torch.nn.Conv2d(4, 8, (2, 2))
    return conv(x)
  
  def bn(self, x, training):
    bn = torch.nn.BatchNorm2d(8)
    return bn(x, training=training)
# Inputs to the model
x = torch.randn(1, 4, 3, 3)
training = True
