
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * -0.1
        v4 = torch.where(v2, v1, v3)
        return v4
negative_slope = 1
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
