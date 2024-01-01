
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.layer = torch.nn.Sequential(self.conv, self.conv)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        o = self.layer(x1)
        x = self.bn(o)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
