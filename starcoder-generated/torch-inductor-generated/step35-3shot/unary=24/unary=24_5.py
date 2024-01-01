
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 512, (7, 20), stride=(2, 6), groups=512, dilation=(1, 1), padding=(3, 10), bias=False)
    def forward(self, x):
        negative_slope = 1e-06
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 1, 512, 512)
