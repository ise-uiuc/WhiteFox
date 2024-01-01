
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v1 = self.bn(v1)
        v2 = self.conv(x2)
        return v1, v2 + v1
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
x2 = torch.randn(1, 3, 3, 3)
