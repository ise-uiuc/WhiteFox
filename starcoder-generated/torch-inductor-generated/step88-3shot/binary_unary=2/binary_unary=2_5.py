
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 64, 1, stride=1, padding=0, groups=8)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0, groups=32)
        self.conv3 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0, groups=16)
        self.conv4 = torch.nn.Conv2d(64, 3, 1, stride=1, padding=0, groups=8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 100.5
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 10000
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 0.5
        v12 = F.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 256, 256, 3)
