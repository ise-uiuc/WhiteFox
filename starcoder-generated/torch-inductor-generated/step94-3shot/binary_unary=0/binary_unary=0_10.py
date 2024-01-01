
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3, groups=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 7, stride=1, padding=3, groups=2)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 + x2
        v5 = torch.relu(v4)
        v6 = v2 + v5
        v7 = torch.relu(v6)
        v8 = self.conv1(x4)
        v9 = self.conv2(v8)
        v10 = self.conv3(v9)
        v11 = v10 + v7
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(16, 16, 64, 64)
x2 = torch.randn(32, 16, 32, 32)
x3 = torch.randn(64, 16, 16, 16)
x4 = torch.randn(16, 16, 64, 64)
