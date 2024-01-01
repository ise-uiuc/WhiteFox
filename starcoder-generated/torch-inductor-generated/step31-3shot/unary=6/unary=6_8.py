
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = (v1 + 3).reshape(2, 2, 8)
        v3 = v1.permute(0, 2, 3, 1) + 3
        v4 = v1 + 3
        v3 = v1.permute(0, 2, 3, 1) + 3
        v4 = v1 + 3
        v4 = v1 + 3
        v4 = v1.permute(0, 2, 3, 1).permute(1, 0, 2, 3) + 3
        v5 = torch.clamp(v1 + 3, min=0, max=6)
        v6 = torch.clamp(v1.permute(0, 2, 3, 1) + 3, min=0, max=6)
        v7 = torch.clamp(v1 + 3, min=0, max=6)
        v8 = torch.clamp(v1.permute(0, 2, 3, 1) + 3, min=0, max=6)
        v9 = torch.clamp(v1 + 3, min=0, max=6)
        v10 = v1 * v4
        v11 = v1 * v4
        v12 = v1 / 6
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
