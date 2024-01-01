
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(23, 34, (5, 5), stride=(2, 1), padding=(1, 2))
    def forward(self, x):
        negative_slope = 0.31919068
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 23, 26, 41)
