
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 2, stride=1, padding=2, dilation=3)
        self.avgpool = torch.nn.AvgPool2d(6, stride=2, padding=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.exp()
        v3 = v2.add(3.0)
        v4 = v3.mul(13)
        v5 = v4.clamp(min=0, max=12)
        v6 = v1 / v5
        v7 = v6.max(dim=3)
        v8 = v7[1].unsqueeze(0)
        v9 = v8.min(dim=2)
        v10 = v9[1].unsqueeze(0)
        v11 = v10.mul(21821)
        v12 = (v11.unsqueeze(-1) * v11.unsqueeze(0)).sum()
        v13 = v12.sqrt()
        return v13
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
