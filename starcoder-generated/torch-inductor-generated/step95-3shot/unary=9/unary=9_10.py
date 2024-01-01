
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2 + 0
        v4 = v3 + 0
        v5 = v4 + 0
        v6 = v5 + 0
        v7 = v6 + 0
        v8 = v7 + 0
        v9 = v8 + 0
        v10 = v2 + 6
        v11 = torch.clamp(v10, min=0)
        v12 = v3 + 6
        v13 = torch.clamp(v12, min=0)
        v14 = v4 + 6
        v15 = torch.clamp(v14, min=0)
        v16 = v5 + 6
        v17 = torch.clamp(v16, min=0)
        v18 = v6 + 6
        v19 = torch.clamp(v18, min=0)
        v20 = v7 + 6
        v21 = torch.clamp(v20, min=0)
        v22 = v8 + 6
        v23 = torch.clamp(v22, min=0)
        v24 = v9 + 6
        v25 = torch.clamp(v24, min=0)
        v26 = v11 + v13
        v27 = v15 + v19
        v28 = v21 + v17
        v29 = v25 + v15
        v30 = v29 + v23
        v31 = v30 / 6
        return v31
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
