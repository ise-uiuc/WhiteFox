
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 1, (5, 5), stride=(3, 5), padding=(4, 1))
    def forward(self, x):
        negative_slope = 0.6197335
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.rand(1, 6, 19, 15)
