
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 9)
        t4 = torch.clamp_max(t3, 18)
        t5 = t1 * t4
        t6 = t5 / 18
        t7 = t6 + t6
        t8 = torch.clamp_min(t7, 0)
        t9 = torch.clamp_max(t8, 6)
        t10 = t1 + t3
        t11 = torch.clamp_min(t10, 0)
        t12 = torch.clamp_max(t11, 6)
        t13 = t4 + t7
        t14 = torch.clamp_min(t13, 0)
        t15 = torch.clamp_max(t14, 6)
        return t12 + t15
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
