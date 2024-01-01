


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(8, 4, 11, stride=1, padding=5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = self.conv2(v6)
        v13 = v11 * v12
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = v18 * 0.5
        return v19
# Inputs to the model
x1 = torch.randn(1, 16, 112, 112)
