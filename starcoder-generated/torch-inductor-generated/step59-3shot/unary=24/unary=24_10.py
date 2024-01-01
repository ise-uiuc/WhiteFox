
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(5, 11, 9, stride=1, padding=3)
        self.conv_2 = torch.nn.Conv2d(11, 16, 16, stride=1, padding=2)
    def forward(self, x):
        negative_slope = 5.645927
        v1 = self.conv_1(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.conv_2(v4)
        v6 = v5 > 0
        v7 = v5 * negative_slope
        v8 = torch.where(v6, v5, v7)
        return v8
# Inputs to the model
x1 = torch.randn(7, 5, 35, 8)
