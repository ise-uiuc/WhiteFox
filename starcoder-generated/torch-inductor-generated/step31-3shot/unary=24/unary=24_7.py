
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = 0.2
        v1 = self.conv_1(x)
        v2 = v1 + x
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        v6 = self.conv_2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
