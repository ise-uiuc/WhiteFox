
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.conv2(s)
        y = self.bn(t)
        return y
# Inputs to the model
x1 = torch.randn(2, 1, 4, 4)
