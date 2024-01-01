
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv(x1)
        v4 = self.conv(x1)
        v5 = self.conv(x1)
        v6 = self.conv(x1)
        v7 = self.conv(x1)
        v8 = self.conv(x1)
        v9 = self.conv(x1)
        v10 = v1 + v2
        v11 = torch.relu(v10)
        v12 = v3 + v4
        v13 = torch.relu(v12)
        v14 = v5 + v6
        v15 = torch.relu(v14)
        v16 = v7 + v8
        v17 = torch.relu(v16)
        v18 = v9 + v11
        v19 = torch.relu(v18)
        v20 = 2 + v13
        v21 = v20 + v15
        v22 = torch.relu(v21)
        v23 = 2 + v17
        v24 = v23 + v19
        v25 = torch.relu(v24)
        v26 = 3 + v9
        v27 = v26 + v22
        v28 = torch.relu(v27)
        v29 = torch.add(v17, v10)
        v30 = torch.relu(v29)
        v31 = torch.add(v22, v23)
        v32 = torch.relu(v31)
        v33 = v32 + v9
        v34 = torch.relu(v33)
        return v34
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
