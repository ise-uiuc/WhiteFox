
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, 1)
        self.bn = torch.nn.BatchNorm2d(64, momentum=None)
    def forward(self, x6):
        x6 = self.conv(x6)
        x6 = self.bn(x6)
        return x6
# Inputs to the model
x6 = torch.randn(1, 3, 224, 224)
