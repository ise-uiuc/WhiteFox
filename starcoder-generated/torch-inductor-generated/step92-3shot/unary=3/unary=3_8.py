
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 1)
        self.conv3 = torch.nn.Conv2d(3, 4, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = v6 + 24
        v8 = self.conv2(v6)
        v9 = torch.sigmoid(v8) + 0.5
        v10 = self.conv3(v6)
        v11 = self.conv2(v10)
        v12 = self.conv1(v11)
        v13 = v12 * 0.5
        v14 = v12 * 0.7071067811865476
        v15 = torch.erf(v14)
        v16 = v15 + 1
        v17 = v13 * v16
        v18 = v13 * 0.7071067811865476
        v19 = torch.erf(v18)
        v20 = v19 + 1
        v21 = v17 * v20
        v22 = self.conv1(x1)
        v23 = v22 * 0.5
        v24 = v22 * 0.7071067811865476
        v25 = torch.erf(v24)
        v26 = v25 + 1
        v27 = v23 * v26
        v28 = self.conv2(v27)
        v29 = v28 * 0.5
        v30 = v28 * 0.7071067811865476
        v31 = torch.erf(v30)
        v32 = v31 + 1
        v33 = v29 * v32
        v34 = self.conv3(v33)
        v35 = self.conv2(v34)
        return v35 + v21
# Inputs to the model
x1 = torch.randn(1, 2, 20, 20)
