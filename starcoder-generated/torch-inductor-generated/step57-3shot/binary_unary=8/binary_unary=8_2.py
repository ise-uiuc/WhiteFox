
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(8, affine=True)
    def forward(self, x1):
        x = self.conv(x1)
        x = self.conv(x)
        x = self.conv(x)
        x = self.conv(x)
        x = self.conv(x)
        x = self.conv(x)
        x1 = self.bn(x)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
