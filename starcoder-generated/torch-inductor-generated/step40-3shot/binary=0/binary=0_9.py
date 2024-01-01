
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=1, bias=None):
        t1 = self.conv(x1)
        b1 = torch.add(t1, bias)
        v2 = b1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
bias = torch.randn(1, 8, 32, 32)
