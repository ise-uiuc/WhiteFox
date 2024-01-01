
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.Sigmoid())
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        s1 = self.conv(x1)
        s1 += x1
        s1 = self.bn(s1)
        return s1
# Inputs to the model
x1 = torch.randn(2, 4, 4, 4)
