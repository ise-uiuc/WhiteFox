
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=True)
        self.bn = torch.nn.BatchNorm2d(3, affine=False)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.bn(s)
        return s
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
