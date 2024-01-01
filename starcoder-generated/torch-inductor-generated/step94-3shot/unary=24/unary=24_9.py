
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 7, stride=(7, 6), dilation=(1, 1), padding=(7, 6), groups=10)
    def forward(self, x):
        negative_slope = 0.9055015
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(40, 1, 92, 94)
