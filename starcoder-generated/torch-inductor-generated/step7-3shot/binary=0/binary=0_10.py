
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        #v2 = self.bn(v1)
        v2 = (v1 + x2) + x3
        return v2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
x2 = torch.randn(5, 3, 64, 64)
x3 = torch.randn(5, 3, 64, 64)
