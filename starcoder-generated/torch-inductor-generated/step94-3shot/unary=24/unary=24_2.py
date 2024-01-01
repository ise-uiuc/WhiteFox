
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d((2, 3, 0, 0))
        self.conv = torch.nn.Conv2d(5, 2, (1, 5), stride=(1, 1), padding=(0, 0))
    def forward(self, x):
        negative_slope = 0.2869379
        v1 = self.pad(x)
        v2 = self.conv(v1)
        v3 = v1 > 0
        v0 = x
        v4 = v1 * negative_slope
        v5 = torch.where(v3, v1, v4)
        v6 = v0 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 5, 7, 8)
