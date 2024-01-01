
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        s1 = self.conv(x1)
        out = self.bn(s1)
        return out
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
