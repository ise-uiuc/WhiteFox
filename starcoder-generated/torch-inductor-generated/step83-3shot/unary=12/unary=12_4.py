
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1, dilation=1, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sqrt()
        v3 = v1.sqrt() * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
