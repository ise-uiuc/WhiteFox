
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
    def forward(self, x1, other=1, x2=None, x3=None, padding1=None):
        var1 = self.conv(x1)
        x2 = x1 + other
        if not padding1 is None:
            var1 += padding1
            x3 = padding1 + other
        out = var1 + x2 + x3
        return out
# Inputs to the model
x0 = torch.randn(1, 1, 64, 64)
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
