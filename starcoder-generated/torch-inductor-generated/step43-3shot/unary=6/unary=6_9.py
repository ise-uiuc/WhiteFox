
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.dequant = torch.quantization.DeQuantStub()
        self.add = torch.nn.quantized.FloatFunctional()
    def forward(self, x1):
        x1 = self.quant(x1)
        x2 = self.conv(x1)
        x3 = torch.clamp_min(x2, 0)
        x4 = torch.clamp_max(x3, 6)
        x5 = self.conv(x1)
        x6 = self.relu(x5)
        x7 = self.dequant(x6)
        x8 = self.add.add(x2, x7)
        x9 = self.add.mul(x8, x4)
        x10 = self.add.div(x9, self.add.scalar(6))
        return x10
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
