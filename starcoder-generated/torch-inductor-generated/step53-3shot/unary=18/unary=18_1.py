
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
    self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
    self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.conv4 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
    self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.conv6 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
    self.maxpool = torch.nn.MaxPool2d(2)

  def forward(self, x1):
    x = self.maxpool(F.relu(self.conv1(x1)))
    x = self.maxpool(F.relu(self.conv2(x)))
    x = self.maxpool(F.relu(self.conv3(x)))
    x = self.maxpool(F.relu(self.conv4(x)))
    x = self.maxpool(F.relu(self.conv5(x)))
    x = self.maxpool(F.relu(self.conv6(x)))
    x = torch.sigmoid(x)
    x = self.maxpool(F.relu(x))
    return x
# Inputs to the model
x1 = torch.rand(1, 3, 64, 64)
