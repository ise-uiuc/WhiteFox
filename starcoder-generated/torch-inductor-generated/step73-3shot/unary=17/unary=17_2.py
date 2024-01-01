
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.deconv1 = nn.ConvTranspose2d(1, 4, 5)
    self.deconv2 = nn.ConvTranspose2d(4, 1, 3)
  def forward(self, x1):
    x1 = F.relu(self.deconv1(x1))
    x1 = self.deconv2(x1)
    return x1
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
