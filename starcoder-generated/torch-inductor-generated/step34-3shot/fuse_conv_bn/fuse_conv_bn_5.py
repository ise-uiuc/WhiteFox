
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x):
        s = self.conv1(x)
        y = self.bn(s)
        return y
# Inputs to the model
x = torch.randn(1, 512, 32, 32)
