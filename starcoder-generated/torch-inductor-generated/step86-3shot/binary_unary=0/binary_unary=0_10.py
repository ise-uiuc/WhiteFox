
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=24)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=32)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = self.conv2(v3)
        v5 = self.conv4(x1)
        v6 = self.conv4(x2)
        v7 = self.conv4(x3)
        v8 = self.conv3(x1)
        v9 = self.conv2(x2)
        v10 = v7 + v8
        v11 = torch.relu(v10)
        v12 = self.conv1(v2)
        v13 = torch.relu(v12)
        v14 = v13 + v9
        v15 = torch.relu(v14)
        v16 = self.conv1(v4)
        v17 = v13 - v16
        v18 = torch.relu(v17)
        return v18
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
