
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - 2.0
        return v2
# Inputs to the model
x3 = torch.randn(1, 1, 1, 1)
