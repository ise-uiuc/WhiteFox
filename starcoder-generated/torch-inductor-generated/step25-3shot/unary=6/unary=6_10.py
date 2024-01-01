
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 24, 2, stride=2, padding=16)
        self.bn = torch.nn.BatchNom2d(123)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(3, 0)
        t4 = torch.clamp_max(3, 6)
        t5 = t1 * t3
        t6 = t5 / 6
        t7 = t7 + t6
        t8 = self.bn(t7)
        return t7
# Inputs to the model
x1 = torch.randn(1, 12, 32, 32)
