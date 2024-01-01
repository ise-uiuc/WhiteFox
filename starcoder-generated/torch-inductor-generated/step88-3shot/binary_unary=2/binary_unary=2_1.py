
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 4, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(4, 16, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 16, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 100
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 100
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 100
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 100
        v12 = F.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 - 100
        v15 = F.relu(v14)
        v16 = self.conv6(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
