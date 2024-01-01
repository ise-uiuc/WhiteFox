
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x3):
        s = self.conv(x3)
        t = self.bn(s)
        return t
# Inputs to the model
x3 = torch.rand(1, 3, 5, 5)
