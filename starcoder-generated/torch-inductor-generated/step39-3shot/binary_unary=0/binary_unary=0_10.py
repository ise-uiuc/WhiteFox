
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1
        v3 = v1 + x2
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v5
        v7 = self.conv2(v6)
        v8 = x3 + v1
        v9 = torch.relu(v8)
        v10 = x2 + v7
        v11 = torch.relu(v10)
        v12 = self.conv1(v11)
        v13 = v12 + v2
        v14 = torch.relu(v13)
        v15 = self.conv3(v14)
        v16 = x1 + x3
        v17 = torch.relu(v16)
        v18 = self.conv3(v17)
        v19 = v18 + v15
        v20 = torch.relu(v19)
        return v20
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
