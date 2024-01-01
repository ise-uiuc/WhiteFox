
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v2 + v1
        v4 = torch.relu(v3)
        a1 = self.conv1(x2)
        a2 = self.conv1(x3)
        v5 = a1 + a2
        v6 = torch.relu(v5)
        v7 = self.conv2(v6)
        v8 = self.conv2(v6)
        v9 = v8 + v7
        v10 = torch.relu(v9)
        v11 = v4 + x4
        v12 = self.conv3(v11)
        v13 = v12 + x1
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
