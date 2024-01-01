
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(149, 175, 2, stride=1, padding=1, dilation=1, groups=1, bias=False)
    def forward(self, x1, other):
        v1 = self.conv1(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 149, 28, 28)
