
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(47, 3, 1, stride=3, padding=6)
        self.linear = torch.nn.Linear(2, 73)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, -5)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        t7 = self.linear(t6)
        t8 = self.linear(t7 + 1)
        t9 = torch.clamp_max(t8, 4)
        t10 = torch.clamp_max(t9, 2)
        t11 = torch.clamp_max(t10, 3)
        t12 = t7 * t11
        t13 = t12 + 1
        t14 = torch.clamp_max(t13, 4.4)
        t15 = torch.clamp_max(t14, 5.5)
        return t15
# Inputs to the model
x1 = torch.randn(1, 47, 40, 32)
