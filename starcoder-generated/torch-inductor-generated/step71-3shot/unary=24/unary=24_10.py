
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 9, 5, stride=1, padding=0)
        self.conv_1 = torch.nn.Conv2d(3, 9, 3, stride=2, padding=0)
        self.conv_2 = torch.nn.Conv2d(9, 9, 3, stride=2, padding=0)
    def forward(self, x):
        negative_slope = -3.9067466
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
x1 = torch.randn(1, 9, 82, 21)
