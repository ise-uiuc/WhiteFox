
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 192, 7, 1, 3)
        self.bn = torch.nn.BatchNorm2d(192)
        self.maxpool = torch.nn.MaxPool2d(2, 2, 1)
    def forward(self, x):
        x1 = self.maxpool(self.bn(self.conv(x)))
        return x1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
