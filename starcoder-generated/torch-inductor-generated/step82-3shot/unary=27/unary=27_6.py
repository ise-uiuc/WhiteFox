
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.quantize = torch.nn.QuantizeStub()
        self.conv = torch.nn.Conv2d(1, 4, 1)
        self.tanh = torch.nn.Tanh()
        self.dequantize = torch.nn.DeQuantize()
    def forward(self, x1):
        v1 = self.quantize(x1)
        v2 = self.conv(v1)
        v3 = self.tanh(v2)
        v4 = self.dequantize(v3)
        return v4
min = 0
max = 0
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6)
