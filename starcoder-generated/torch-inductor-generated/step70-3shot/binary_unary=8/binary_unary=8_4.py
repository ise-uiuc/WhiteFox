
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 11, stride=5, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        y = self.conv(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
