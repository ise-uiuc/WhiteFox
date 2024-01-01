
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 10, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(10, 10, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(10, 10, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + 0.4
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + 1
        v6 = torch.relu(v5)
        v7 = self.conv3(v2)
        v8 = v7 + 2
        v9 = torch.tanh(v5)
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v11 + 3.4
        v13 = self.conv3(v10)
        v14 = v13 + 3.4
        v15 = self.conv3(v14)
        v16 = v15 + 4
        v17 = self.conv3(x1)
        v18 = v17 + 4
        v19 = self.conv3(v18)
        v20 = v19 + 16
        v21 = self.conv3(x1)
        v22 = self.conv3(v21)
        v23 = v22 + 4
        v24 = v23 + 4
        v25 = self.conv3(v24)
        v26 = torch.relu(v25)
        return v26
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
