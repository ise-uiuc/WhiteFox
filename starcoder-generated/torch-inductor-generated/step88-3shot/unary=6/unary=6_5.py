
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=1, padding=1)
        self.b1 = torch.nn.BatchNorm2d(3)
        self.b2 = torch.nn.BatchNorm2d(3)
        self.b3 = torch.nn.BatchNorm2d(3)
        self.b4 = torch.nn.BatchNorm2d(3)
        self.b5 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = self.b1(t5)
        t7 = self.b2(t5)
        t8 = self.b3(t5)
        t9 = self.b4(t5)
        t10 = self.b5(t5)
        t11 = t6 + t7 + t8 + t9 + t10
        return t11
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
