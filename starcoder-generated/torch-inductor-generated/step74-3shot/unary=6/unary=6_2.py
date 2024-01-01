
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 15, 2, stride=2, padding=4)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0)
        t4 = torch.clamp(t3, min=3)
        t5 = t1 * t3
        t6 = t5 / 6
        t7 = self.conv2(t6)
        return t7
x1 = torch.randn(4, 3, 35, 35)
