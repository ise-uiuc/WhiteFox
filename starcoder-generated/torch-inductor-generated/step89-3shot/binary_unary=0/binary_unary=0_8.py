
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.depthwise = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x2)
        v2 = self.depthwise(x2)
        v3 = v1 + x2
        v4 = torch.relu(v3)
        v5 = v4 + v2
        v6 = torch.relu(v5)
        v7 = self.conv2(v6)
        v8 = v7 + v4
        v9 = torch.relu(v8)
        v10 = self.conv3(v9)
        v11 = v10 + x2
        v12 = torch.relu(v11)
        v13 = self.conv3(v12)
        v14 = v13 + x3
        v15 = torch.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
