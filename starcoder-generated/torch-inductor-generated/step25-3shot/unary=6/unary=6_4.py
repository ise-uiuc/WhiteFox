
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        return t5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
