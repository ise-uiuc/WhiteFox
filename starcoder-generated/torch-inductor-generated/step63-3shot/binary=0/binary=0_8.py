
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x0, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22):
        v1 = self.conv(x1)
        v2 = v1 + x1
        v3 = v2 + x2
        v4 = v3 + x3
        v5 = v4 + x4
        v6 = v5 + x5
        v7 = v6 + x6
        v8 = v7 + x7
        v9 = v8 + x8
        v10 = v9 + x9
        v11 = v10 + x11
        v12 = v11 + x12
        v13 = v12 + x13
        v14 = v13 + x14
        v15 = v14 + x15
        v16 = v15 + x16
        v17 = v16 + x17
        v18 = v17 + x18
        v19 = v18 + x19
        v20 = v19 + x20
        v21 = v20 + x21
        v22 = v21 + x22
        v23 = v22 + v12
        v24 = v23 + x5
        v25 = v24 + torch.randn(v1.shape)
        return v25
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
