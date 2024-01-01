
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.add = torch.nn.quantized.FloatFunctional()
        self.div = torch.div
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = self.add.add_scalar(v2, 3.0)
        v4 = v3.clamp(min=0.0, max=6.0)
        v5 = self.div(v4, 6.0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
