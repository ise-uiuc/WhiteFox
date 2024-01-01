
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 9)
        self.conv3 = torch.nn.Conv2d(16, 32, 7)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp(t2, 0, 6)
        t4 = t3 / 6
        t5 = self.conv2(t4)
        t6 = t5 + 3
        t7 = torch.clamp(t6, 0, 6)
        t8 = t7 / 6
        t9 = self.conv3(t8)
        t10 = t9 + 3
        t11 = torch.clamp(t10, 0, 6)
        t12 = t11 / 6
        return t12
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
