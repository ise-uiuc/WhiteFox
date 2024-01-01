
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.linear1 = torch.nn.Linear(1, 1)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + x2
        v4 = torch.relu(v3)
        v5 = v2 + v4
        v6 = torch.relu(v5)
        v7 = x1 + v6
        v8 = torch.nn.ReLU()(v7)
        v9 = self.conv3(v8)
        v10 = v9 + x4
        v11 = torch.nn.ReLU()(v10)
        v12 = v11 + x5
        v13 = torch.nn.ReLU()(v12)
        v14 = v13.reshape(1, -1)
        v15 = self.linear1(v14)
        v16 = v15 + v13
        v17 = torch.relu(v16)
        v18 = v17.reshape(shape=(-1, 1, 6, 6))
        return v18
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
