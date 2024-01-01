
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 3)
        self.bn = torch.nn.BatchNorm2d(24)
        self.pool = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.pool(self.bn(self.conv(x)))
        return x
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
