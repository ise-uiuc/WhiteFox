
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 * 0.5
        t3 = torch.clamp(t2 + 3, 0, 6)
        t4 = t1 * t3
        t5 = t3 * 0.5
        t6 = t4 - t5
        t7 = t6 + 3
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
