
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=0, dilation=2, groups=2)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = v1 - False
        return v2
# Inputs to the model
x4 = torch.randn(1, 3, 2, 2)
