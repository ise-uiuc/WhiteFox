
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 5, stride=1, padding=2, dilation=1, groups=1, bias=False)
        self.bias = torch.nn.Parameter(torch.randn(32, 64, 64))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.bias
        v3 = v1 - v2
        return v3
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
