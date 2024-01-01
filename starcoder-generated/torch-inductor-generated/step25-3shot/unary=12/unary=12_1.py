
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 56, stride=32, padding=2, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.tanh()
        v3 = v2.sigmoid()
        v4 = v1 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
