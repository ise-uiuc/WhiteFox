
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        v1 = self.bn(self.conv(x1))
        v2 = self.bn(self.conv(x2))
        return (torch.cat((v1, v2,), 1),)
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
x2 = torch.randn(1, 2, 4, 4)
