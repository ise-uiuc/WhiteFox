
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=3, padding=1)
        self.conv1 = torch.nn.Conv2d(4, 4, 2, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.conv1(t1)
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        return t5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
