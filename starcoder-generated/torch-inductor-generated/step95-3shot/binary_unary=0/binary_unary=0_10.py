
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(x2)
        v5 = 1 + v4
        v6 = torch.relu(v5)
        v7 = 2 + self.conv3(x3)
        v8 = torch.relu(v7)
        v9 = v8 + self.conv1(x4)
        v10 = torch.relu(v9)
        v11 = v10 + self.conv2(x5)
        v12 = torch.relu(v11)
        v13 = 3 + self.conv3(x6)
        v14 = torch.relu(v13)
        v15 = self.conv1(v14)
        v16 = 2 * v15
        v17 = torch.relu(v16)
        v18 = self.conv2(v17)
        v19 = 1 * v18
        v20 = torch.relu(v19)
        v21 = self.conv3(v20)
        v22 = 1 * v21
        v23 = torch.relu(v22)
        return v23
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
