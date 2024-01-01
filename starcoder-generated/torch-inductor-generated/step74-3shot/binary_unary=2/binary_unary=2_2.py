
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(22, 24, 1, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(24, 32, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=7)
        self.conv4 = torch.nn.Conv2d(64, 80, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(80, 192, 1, stride=1, padding=9)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1000
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 4000
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 1100
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 5000
        v12 = F.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 - 9000
        v15 = F.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 22, 64, 64)
