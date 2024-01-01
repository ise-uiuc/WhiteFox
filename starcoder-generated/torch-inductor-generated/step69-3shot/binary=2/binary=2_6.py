
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 10, 1, stride=1, padding=0, dilation=2, groups=1, bias=True)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - 200000.1
        return v2
# Inputs to the model
x2 = torch.randn(2, 2, 6, 6)
