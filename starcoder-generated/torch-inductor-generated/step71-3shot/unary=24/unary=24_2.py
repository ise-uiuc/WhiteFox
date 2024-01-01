
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 7, 3, stride=1, padding=3)
        self.conv_1 = torch.nn.Conv2d(7, 5, 1, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(5, 3, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.8643128
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv_1(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        v9 = self.conv_2(v8)
        v10 = v9 > 0
        v11 = v9 * negative_slope
        v12 = torch.where(v10, v9, v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 7, 12, 58)
