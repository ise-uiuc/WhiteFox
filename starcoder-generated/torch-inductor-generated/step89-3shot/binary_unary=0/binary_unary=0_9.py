
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 2, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.relu(v3)
        v5 = torch.cat([v4, x2], dim=1)
        v6 = self.conv1(v5)
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = torch.relu(v8)
        v10 = torch.cat([v9, x2], dim=1)
        v11 = self.conv1(v10)
        v12 = self.conv2(v11)
        v13 = self.conv3(v12)
        v14 = torch.relu(v13)
        return torch.cat([v14, x2], dim=1)
# Inputs to the model
x1 = torch.randn(1, 3, 48, 48)
x2 = torch.randn(1, 128, 24, 24)
