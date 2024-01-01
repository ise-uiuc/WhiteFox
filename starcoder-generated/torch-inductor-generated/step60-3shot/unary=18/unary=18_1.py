
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=(15, 21), stride=1, padding=2, bias=False)
        self.bn = torch.nn.BatchNorm2d(2)
        self.avgpool = torch.nn.AvgPool2d((1, 13), (1, 12))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.avgpool(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 56, 84)
