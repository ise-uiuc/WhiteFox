
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = torch.clamp(t1 + 3, 0, 6)
        t3 = t1 * t2
        t4 = t3 / 6
        return t4
# Inputs to the model
x1 =torch.randn(1, 3, 64, 64)
