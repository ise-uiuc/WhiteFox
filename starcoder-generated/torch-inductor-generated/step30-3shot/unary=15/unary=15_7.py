
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.conv5 = torch.nn.ConvTranspose2d(32, 1, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.relu(self.bn1(self.conv1(x1)))
        v2 = torch.relu(self.bn2(self.conv2(v1)))
        v3 = torch.relu(self.bn3(self.conv3(v2)))
        v4 = self.bn4(self.conv4(v3))
        v5 = torch.relu(v4)
        v6 = torch.sigmoid(self.conv5(v5))
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 256, 256)
