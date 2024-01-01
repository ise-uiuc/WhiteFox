
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.relu(v1)
        v4 = self.conv2(v3)
        v5 = v4 + v2
        v6 = self.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + x3
        v9 = self.relu(v8)
        v10 = self.conv2(v8)
        v11 = self.relu(v10)
        v12 = v11 + x4
        v13 = self.relu(v12)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
