
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=16)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=16)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + v1
        v6 = torch.relu(v5)
        v7 = v3 + v6
        v8 = torch.relu(v7)
        v9 = v3 + v8
        v10 = torch.relu(v9)
        v11 = v3 + v10
        v12 = torch.relu(v11)
        v13 = v3 + v12
        v14 = torch.relu(v13)
        v15 = v3 + v14
        v16 = torch.relu(v15)
        v17 = v16.view(x1.size())
        v18 = v17 - x1
        v19 = self.conv1(v18)
        v20 = 2 + v19
        v21 = torch.relu(v20)
        return v21 # This model satisfies the pattern, and can be optimized by XLA.
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
