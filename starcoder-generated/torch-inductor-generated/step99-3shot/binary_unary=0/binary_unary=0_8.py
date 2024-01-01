
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x1
        v6 = torch.relu(v5)
        v7 = v2 + v5
        v8 = torch.relu(v7)
        v9 = self.conv3(v6)
        v10 = v9 + x1
        v11 = torch.relu(v10)
        v12 = v9 + v10
        v13 = torch.relu(v12)
        v14 = self.conv4(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
