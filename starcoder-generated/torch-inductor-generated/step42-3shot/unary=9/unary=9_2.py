
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.add = torch.nn.quantized.FloatFunctional()
        self.clamp = torch.nn.quantized.FloatFunctional()
        self.div = torch.nn.quantized.FloatFunctional()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.add.add_scalar(v1, 3)
        v3 = self.clamp.clamp(v2, 0, 6)
        v4 = self.div.div(v3, 6)
        return v4
# Inputs to the model
x1 = torch.randn(8, 3, 64, 64)
