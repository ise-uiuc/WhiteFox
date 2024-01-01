
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(63, 101, 7, stride=(3, 2), padding=(6, 2), groups=12)
    def forward(self, x):
        negative_slope = 0.9181319
        v1 = self.conv2d(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(3, 63, 98, 171)
