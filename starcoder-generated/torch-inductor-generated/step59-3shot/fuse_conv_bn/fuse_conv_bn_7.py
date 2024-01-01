
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, 2)
        self.bn = torch.nn.BatchNorm2d(4, momentum=0.0, affine=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
