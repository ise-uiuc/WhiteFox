
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 5, stride=1, padding=0, dilation=1, groups=3, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.09766342
        return v2
# Inputs to the model
x = torch.randn(1, 1, 112, 112)
