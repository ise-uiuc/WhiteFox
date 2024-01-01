
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x2):
        s = self.conv(x2)
        z = self.bn(s)
        return s
# Inputs to model
x2 = torch.randn(1, 2, 6, 6)
