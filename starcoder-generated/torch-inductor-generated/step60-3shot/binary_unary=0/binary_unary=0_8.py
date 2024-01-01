
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(32, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = self.conv3(x2)
        v5 = self.conv1(v4)
        v6 = torch.relu(v5)
        a1 = self.conv2(v6)
        a2 = self.conv1(v6)
        v7 = v1 + a1
        v8 = torch.relu(v7)
        v9 = self.conv1(v8)
        v10 = self.conv2(v9)
        v11 = v10 + x5
        v12 = torch.relu(v11)
        v13 = a2 + x4
        v14 = torch.relu(v13)
        v15 = self.conv2(v12)
        v16 = v15 + x3
        v17 = torch.relu(v16)
        return v17
# Inputs to the model
b1 = torch.randn(1, 32, 64, 64)
b2 = torch.randn(1, 32, 64, 64)
b3 = torch.randn(1, 32, 64, 64)
b4 = torch.randn(1, 32, 64, 64)
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 32, 64, 64)
x4 = torch.randn(1, 32, 64, 64)
x5 = torch.randn(1, 32, 64, 64)
