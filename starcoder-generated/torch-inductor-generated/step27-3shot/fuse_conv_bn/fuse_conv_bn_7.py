
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x3):
        v = self.conv(x3)
        v = self.bn(v)
        return v
# Inputs to the model
x3 = torch.randn(1, 3, 3, 3)
