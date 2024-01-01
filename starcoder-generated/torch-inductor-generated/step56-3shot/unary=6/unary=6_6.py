
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 115, 3, stride=2, padding=3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t1_2 = self.conv(t1)
        t2 = t1_2 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        return t6
x1 = torch.randn(2, 3, 28, 28)
