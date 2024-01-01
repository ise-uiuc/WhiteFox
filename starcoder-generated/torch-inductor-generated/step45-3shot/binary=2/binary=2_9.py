
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 64, 5, stride=3, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0
        return v2
# Inputs to the model
x = torch.randn(1, 8, 64, 64)
