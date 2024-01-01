
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 48, 2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(48, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = v12 * 0.5
        v14 = v12 * 0.7071067811865476
        v15 = torch.erf(v14)
        v16 = v15 + 1
        v17 = v13 * v16
        v18 = v17 * 0.5
        v19 = self.conv3(v6)
        return v19
# Inputs to the model
x1 = torch.randn(1, 32, 56, 56)
