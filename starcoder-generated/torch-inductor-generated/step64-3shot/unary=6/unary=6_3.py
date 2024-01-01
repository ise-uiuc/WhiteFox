
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = self.conv2(t5)
        t7 = torch.mean(t6, dim=1)
        t8 = t7 + 3
        t9 = torch.clamp(t8, 0, 6)
        t10 = t1.mul(t9)
        t11 = t10 / 6
        t12 = torch.mean(t11, dim=1)
        t13 = self.conv3(t12)
        t14 = self.conv4(t13)
        t15 = torch.mean(t14, dim=1)
        return t15
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
