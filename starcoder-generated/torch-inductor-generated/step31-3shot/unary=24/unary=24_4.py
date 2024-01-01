
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(24, 24, 1, stride=2, padding=0)
        self.conv_2 = torch.nn.Conv2d(24, 24, 1, stride=2, padding=0)
    def forward(self, x):
        negative_slope = 0.2
        v1 = self.conv_1(x)
        v2 = v1 * negative_slope
        v3 = v2 % 1
        v4 = self.conv_2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 24, 32, 32)
