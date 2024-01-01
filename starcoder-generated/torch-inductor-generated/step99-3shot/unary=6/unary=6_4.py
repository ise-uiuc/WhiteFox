
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(32, 4, 1, stride=1, padding=1, groups=1)
        self.conv_b = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1, groups=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        t1 = self.conv_a(x1)
        t2 = self.conv_b(t1)
        t3 = self.bn(t2)
        return t3
# Inputs to the model
x1 = torch.randn(3, 32, 56, 64)
