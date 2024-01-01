
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3, stride=3, padding=1)
        self.pooling = torch.nn.AvgPool2d((3, 3), stride=(3, 3), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = torch.clamp_min(v6, 0)
        v8 = torch.clamp_max(v7, 6)
        t1 = self.pooling(v8)
        v9 = t1 + 3
        v10 = torch.clamp_min(v9, 0)
        v11 = torch.clamp_max(v10, 6)
        v12 = v8 * v11
        v13 = v12 / 6
        v14 = torch.clamp_min(v13, 0)
        v15 = torch.clamp_max(v14, 6)
        v16 = v13 * v15
        v17 = v16 / 6
        v18 = t1 * v17
        v19 = v18 / 6
        return v19
# Inputs to the model
x1 = torch.randn(2, 2, 64, 64)
