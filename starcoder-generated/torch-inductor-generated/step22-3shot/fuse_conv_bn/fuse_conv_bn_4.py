
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 32, 3)
        self.bn = torch.nn.BatchNorm2d(32)
        self.c2 = torch.nn.Conv2d(32, 32, 3)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((32, 32))
    def forward(self, x):
        x = self.c1(x)
        x = self.bn(x)
        x = self.c2(x)
        x = self.bn2(x)
        return self.avgpool(x)
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
