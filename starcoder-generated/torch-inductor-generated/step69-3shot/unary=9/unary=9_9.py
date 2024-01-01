
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = v2.clamp(min=0)
        v4 = v3.clamp(max=6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(5, 8, 64, 64)
