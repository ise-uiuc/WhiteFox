
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 5, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 * 3
        t3 = t2 + 2
        t4 = t3 + 2
        t5 = t2 - 2
        t6 = t2 * t4
        t7 = t5 * t6
        t8 = t7 / 2
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
