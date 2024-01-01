
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        x3 = self.bn(self.conv(x1))
        x4 = self.bn(self.conv(x2))
        return x3, x4
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
x2 = torch.randn(1, 3, 5, 5)
