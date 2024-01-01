
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 1, 7, stride=1, padding=3)
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
        v19 = self.conv3(v10)
        v20 = v18 * v19
        v21 = v20 * 0.5
        v22 = v20 * 0.7071067811865476
        v23 = torch.erf(v22)
        v24 = v23 + 1
        v25 = v21 * v24
        v26 = v25 * 0.5
        return v25
# Inputs to the model
x1 = torch.randn(1, 1, 39, 39)
