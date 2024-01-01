
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x4):
        x4 = self.conv(x4)
        y4 = self.bn(x4)
        x4 = self.bn(x4)
        return x4 + y4
# Inputs to the model
x4 = torch.randn(1, 3, 4, 4)
