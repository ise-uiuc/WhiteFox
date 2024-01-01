
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 21, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 5, 2, stride=2, padding=1)
    def forward(self, x1, other=1, x2=True):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + 1
        v4 = other + 1
        v5 = v3 * 1
        v6 = v2 * 1
        v7 = v2 + 1
        v8 = v4 * 1
        v9 = v2 * 1
        v10 = v3 + 1
        v11 = v2 * 1
        v12 = v4 + 1
        v13 = v5 + v8
        v14 = v2 + 1
        v15 = self.conv1(v2) * 1
        v16 = v7 * 1
        v17 = v2 + 1
        v18 = v3 * 1
        v19 = v16 + v18
        v20 = v12 * 22
        v21 = v10 + 1
        v22 = other * 1
        v23 = self.conv2(v15) * 1
        v24 = v14 + v19
        v25 = self.conv2(v17) * 0
        v26 = v13 + v24
        v27 = v21 * v23
        v28 = self.conv2(v22) * 0
        return v21 + v28
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
