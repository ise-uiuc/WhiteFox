
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = torch.clamp(t2, 0, 6)
        t4 = 3 + t1
        t5 = torch.clamp(t4, 0, 6)
        t6 = t1 * t3
        t7 = t6 / 6
        t8 = t1 * t5
        t9 = t8 / 6
        return t7 + t9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
