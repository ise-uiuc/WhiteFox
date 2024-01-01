
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 5, 5, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(5)
        self.conv_bn = nn.Conv2d(10, 5, 5, stride=1, padding=1, bias=False)
        self.bn_conv = nn.BatchNorm2d(5)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x1 = self.conv_bn(x)
        x2 = self.bn_conv(x)
        return x1 + x2
# Inputs to the model
x = torch.randn(2, 10, 4, 4)
