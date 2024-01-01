
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = v1 + x5
        v3 = torch.relu(v2)
        a1 = self.conv1(x1)
        a2 = torch.relu(a1)
        a3 = self.conv2(a2)
        a4 = torch.relu(a3)
        a5 = self.conv3(v3)
        a6 = torch.relu(a5)
        v4 = a4 + a6
        v5 = self.conv2(v4)
        v6 = v5 + x4
        v7 = torch.relu(v6)
        v8 = self.conv3(v7)
        v9 = v8 + x3
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v11 + x2
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
