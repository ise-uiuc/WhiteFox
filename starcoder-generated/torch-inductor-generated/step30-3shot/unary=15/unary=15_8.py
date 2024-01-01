
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 5, stride=2, padding=0, dilation=1)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0, dilation=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=2, dilation=2)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(128, 512, 1, stride=1, padding=1, dilation=1)
        self.bn4 = torch.nn.BatchNorm2d(512)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.bn1(self.conv2(v2))
        v4 = torch.relu(v3)
        v5 = self.bn2(self.conv3(v4))
        v6 = torch.relu(v5)
        v7 = self.bn3(self.conv4(v6))
        v8 = torch.relu(v7)
        v9 = self.bn4(self.conv5(v8))
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
