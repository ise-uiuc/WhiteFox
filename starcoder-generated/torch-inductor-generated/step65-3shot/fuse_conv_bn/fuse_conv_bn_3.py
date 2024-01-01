
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.conv = torch.nn.Conv2d(1, 3, 3)
        torch.manual_seed(7)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        r1 = self.bn(self.conv(x1))
        r2 = self.bn(self.conv(r1))
        return r2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
