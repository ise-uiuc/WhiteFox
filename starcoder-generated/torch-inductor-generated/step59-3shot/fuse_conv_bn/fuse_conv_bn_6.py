
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 2)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool2d = torch.nn.MaxPool2d(2)
    def forward(self, x4):
        x4 = self.conv(x4)
        x4 = self.bn(x4)
        x4 = self.pool2d(x4)
        return x4
# Inputs to the model
x4 = torch.randn(2, 3, 10, 10)
