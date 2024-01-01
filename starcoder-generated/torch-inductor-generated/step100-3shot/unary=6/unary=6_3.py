
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=3, padding=1, dilation=2)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 / 2
        t3 = torch.clamp(t2, min=0, max=6)
        t4 = torch.add(t1, 2)
        t5 = t4 * t3
        t6 = t5 * 6.0
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
