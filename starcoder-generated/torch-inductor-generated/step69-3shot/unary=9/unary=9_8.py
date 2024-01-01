
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        t0 = 3
        t1 = v1.add(t0)
        v2 = t1.clamp(min=0, max=6)
        t2 = v2 / 6
        return t2
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
