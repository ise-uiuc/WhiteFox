
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        t0 = 3
        v2 = v1 + t0
        t1 = v2.clamp(0, 6)
        v3 = v1 + 3
        t2 = t1 / 6
        return t2
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
