
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 7)
        self.bn = torch.nn.BatchNorm2d(3, affine=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 1024)
