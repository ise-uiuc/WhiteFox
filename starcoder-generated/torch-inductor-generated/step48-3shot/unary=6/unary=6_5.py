
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x1):
        a1 = self.conv(x1)
        return a1
# Inputs to the model
x1 = torch.randn(1, 2, 224, 224)
