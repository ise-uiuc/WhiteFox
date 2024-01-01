
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(26, 98, 3, stride=1, padding=2)
    def forward(self, x, y):
        negative_slope = -2.3739254
        v1 = x + y
        v2 = self.conv(v1)
        v3 = v2 > 0
        v4 = v2 * negative_slope
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
input_1 = torch.randn(22, 26, 6, 30)
input_2 = torch.randn(22, 98, 6, 30)
