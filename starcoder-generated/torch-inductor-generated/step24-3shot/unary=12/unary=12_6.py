
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(72, 13, 63, stride=8, padding=16, dilation=9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.tanh()
        v3 = v2.mul(v1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 72, 16, 16)
