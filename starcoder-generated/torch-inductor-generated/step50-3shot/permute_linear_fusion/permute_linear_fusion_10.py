
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, groups=1, bias=True)
        self.add = torch.nn.quantized.FloatFunctional()
    def forward(self, x1):
        x1 = torch.rand(1, 2, 3, 3)
        x1 = self.conv(x1)
        x1 = self.add.mul(x1, x1 * 2)
        x1 = self.add.mul(x1, x1 * 2)
        x1 = self.add.mul(x1, x1 * 2)
        x1 = self.add.mul(x1, x1 * 2)
        v2 = x1.to(torch.int64)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
