
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, t0):
        t1 = self.conv(t0)
        t2 = t1 + 3
        t3 = t2.clamp(min=0, max=6)
        t4 = t3 / 6
        return t4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
