
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.bn(self.conv(x1))
        v2 = self.conv(x2)
        v3 = torch.nn.functional.conv2d(v2, weight=self.conv.weight, bias=None, stride=1, padding=1, dilation=1, groups=1)
        v4 = v3 + v1
        return torch.nn.ReLU(inplace=True)(v4)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
