
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 9, stride=(3, 3), padding=(3, 3), dilation=(2, 2), groups=3, bias=True)
    def forward(self, x):
        negative_slope = -2.093506
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
