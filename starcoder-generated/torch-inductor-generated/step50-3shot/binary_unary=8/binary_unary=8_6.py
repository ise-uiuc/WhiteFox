
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv1(x1)
        v6 = self.conv1(x1)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        v9 = self.conv1(x1)
        v10 = self.conv1(x1)
        v11 = v9 + v10
        v12 = torch.relu(v11)
        v13 = v4 + v8 + v12
        v14 = self.conv2(x1)
        v15 = self.conv2(x1)
        v16 = v14 + v15
        v17 = torch.relu(v16)
        v18 = self.conv2(x1)
        v19 = self.conv2(x1)
        v20 = v18 + v19
        v21 = torch.relu(v20)
        v22 = v17 + v21
        return v13 + v22
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
