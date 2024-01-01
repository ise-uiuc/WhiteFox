
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1)
        self.conv2 = torch.nn.Conv2d(16, 16, 1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.conv2(t1)
        t3 = t2 + 3
        t4 = torch.clamp(t3, 0, 6)
        t5 = torch.cat((t4, t5), 1)
        t6 = t5 * t1
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
x2 = torch.randn(2, 3, 64, 64)
