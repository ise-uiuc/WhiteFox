
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = self.conv1(x3)
        v4 = v1 + x4
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + x1
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v9 + v2
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
