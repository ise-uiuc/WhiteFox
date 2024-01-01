
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.4
        v3 = v1 * 0.5
        v4 = v1 * 0.6
        v5 = v1 * 0.7071067811865476
        v6 = torch.erf(v1)
        v7 = v6 + 1
        v8 = v2 * v7
        v9 = v3 * v7
        v10 = v4 * v7
        v11 = v5 * v7
        v12 = v1 * v7
        v13 = v11 + v1
        v14 = v13 * 0.4
        v15 = v13 * 0.5
        v16 = v13 * 0.6
        v17 = v13 * 0.7071067811865476
        v18 = torch.erf(v13)
        v19 = v18 + 1
        v20 = v14 * v19
        v21 = v15 * v19
        v22 = v16 * v19
        v23 = v17 * v19
        v24 = v13 * v19
        v25 = v24 + 1
        v26 = v20 * v25
        v27 = v21 * v25
        v28 = v22 * v25
        v29 = v23 * v25
        v30 = self.conv(v29)
        return v30
# Inputs to the model
x1 = torch.randn(1, 1, 195, 216)
