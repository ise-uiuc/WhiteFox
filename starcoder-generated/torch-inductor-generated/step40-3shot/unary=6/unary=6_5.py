
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t1, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t4 * t2
        t6 = t5 / 6
        t7 = self.conv2(x1)
        t8 = t7 + 3
        t9 = torch.clamp_min(t6, 0)
        t10 = torch.clamp_max(t9, 6)
        t11 = t4 * t8
        t12 = t11 / 6
        return t12
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
