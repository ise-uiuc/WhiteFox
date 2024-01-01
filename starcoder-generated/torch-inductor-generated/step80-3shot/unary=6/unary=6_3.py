
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.bn(t1)
        t3 = t1 + 3
        t4 = torch.clamp(t3, 0, 6)
        t5 = t2 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(32, 3, 64, 64)
