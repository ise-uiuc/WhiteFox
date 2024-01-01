
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 24, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 72, 5, stride=2, padding=2)
    def forward(self, x1, x2):
        v1 = x1
        v2 = self.conv1(v1)
        v3 = v2 + 1
        v4 = torch.relu(v3)
        v5 = v4
        v6 = self.conv2(v5)
        v7 = v6
        v8 = torch.relu(v7)
        v9 = v8
        v10 = self.conv3(v9)
        v11 = x2
        v12 = v11
        v13 = self.conv1(v12)
        v14 = v10 + v13
        v15 = torch.relu(v14)
        v16 = v15
        v17 = self.conv2(v16)
        v18 = v17
        v19 = torch.relu(v18)
        v20 = v19
        v21 = self.conv3(v20)
        v22 = v21
        v23 = v22 + 1
        v24 = torch.relu(v23)
        v25 = v24
        v26 = v25 + 1
        v27 = torch.relu(v26)
        return v27
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
