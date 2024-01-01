
class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
    self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
    z = torch.randn(4, 4)
    z2 = torch.randn(1, 4, 4, 4)
    self.conv2(self.conv1(z.view(1, 1, 4, 4)))
    self.conv2(self.conv1(z2))
# Inputs to the model
torch.randn(4, 5, 1, 28, 28)
