
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 10, (4, 7), stride=(2, 2), padding=(3, 0))
    def forward(self, x):
        negative_slope = -2.579492
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(4, 8, 9, 10)
