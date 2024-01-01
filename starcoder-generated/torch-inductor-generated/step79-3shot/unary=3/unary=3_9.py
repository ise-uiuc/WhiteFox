
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 128, (1, 3), stride=1, padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(128, 64, (7, 1), stride=2, padding=(3, 0))
        self.conv3 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 * 0.5
        t3 = t1 * 9.194426955240847e-07
        t4 = torch.asin(t3)
        t5 = t4 + 1
        t6 = t2 * t5
        t7 = self.conv2(t6)
        t8 = t7 * 0.5
        t9 = t7 * -2.424600365610833e-06
        t10 = torch.asin(t9)
        t11 = t10 + 1
        t12 = t8 * t11
        t13 = self.conv3(t12)
        return t13
# Inputs to the model
x1 = torch.randn(1, 32, 37, 49)
