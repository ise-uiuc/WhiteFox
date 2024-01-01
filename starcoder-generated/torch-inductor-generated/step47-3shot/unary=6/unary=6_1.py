
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1.abs()
        t3 = t1 - t2
        t4 = t1.add(t3, alpha=3)
        t5 = torch.clamp(t4, 0, 6)
        t6 = 4 * t5
        t7 = t1 - t6
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
