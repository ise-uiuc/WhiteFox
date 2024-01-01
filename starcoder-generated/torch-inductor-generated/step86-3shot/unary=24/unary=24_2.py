
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, (3, 4), stride=(3, 2), padding=(6, 1), groups=1, bias=True)
    def forward(self, x):
        negative_slope = 0.02315218
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.rand(1, 16, 41, 35)
