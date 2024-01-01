
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(3)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((3,3))
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        y = self.avg_pool(x)
        return y
# Inputs to the model
x = torch.randn(1, 3, 8, 8)
