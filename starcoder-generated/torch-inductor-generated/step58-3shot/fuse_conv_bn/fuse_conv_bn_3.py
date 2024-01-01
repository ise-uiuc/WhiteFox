
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, momentum=0.9, affine=True)
        self.max_pool2d = torch.nn.MaxPool2d(1)
    def forward(self, x):
        z = self.conv(x)
        z = self.bn(z)
        z = self.max_pool2d(z)
        return (z + 2)
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
