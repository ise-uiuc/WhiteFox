
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 20, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(20, 128, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(256)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - torch.rand(1, 20, 96, 96)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - torch.rand(1, 128, 46, 46)
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - torch.rand(1, 256, 46, 46)
        v9 = self.bn(v8)
        v10 = F.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 7, 96, 96)
