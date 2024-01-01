
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 24, 3, stride=2, padding=1, dilation=1, groups=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 6.0
        return v2
# Inputs to the model
x = torch.randn(1, 12, 100, 50)
