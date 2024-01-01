
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.negative_slope = 0.1
    self.fc = torch.nn.Linear(128, 64)
    self.conv = torch.nn.Conv2d(128, 128, 2, stride=1, padding=0)
    self.fc1 = torch.nn.Linear(64, 32)
    self.conv1 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
    self.conv2 = torch.nn.Conv2d(64, 128, 2, stride=1, padding=0)
    self.fc2 = torch.nn.Linear(128, 64)
    self.conv3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
    self.conv4 = torch.nn.Conv2d(128, 256, 5, stride=1, padding=0)
    self.fc3 = torch.nn.Linear(256, 128)
    self.conv5 = torch.nn.Conv2d(128, 32, 1, stride=1, padding=0)
    self.conv6 = torch.nn.Conv2d(32, 64, 5, stride=1, padding=0)
  def forward(self, x1):
    v0 = self.fc(x1)
    v1 = self.conv(v0)
    v2 = torch.max(v1, 3)[0].max(2)[0]
    v3 = F.leaky_relu(v2, self.negative_slope)
    v4 = self.fc1(v3)
    v5 = self.conv1(v4)
    v6 = self.conv2(v5)
    v7 = torch.max(v6, 3)[0].max(2)[0]
    v8 = F.leaky_relu(v7, self.negative_slope)
    v9 = self.fc2(v8)
    v10 = self.conv3(v9)
    v11 = self.conv4(v10)
    v12 = torch.max(v11, 3)[0].max(2)[0]
    v13 = F.leaky_relu(v12, self.negative_slope)
    v14 = self.fc3(v13)
    v15 = self.conv5(v14)
    v16 = self.conv6(v15)
    return v16
# Inputs to the model
x1 = torch.randn(1, 128)
