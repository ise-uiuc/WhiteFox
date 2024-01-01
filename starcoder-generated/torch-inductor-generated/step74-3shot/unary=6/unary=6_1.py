
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 13, 3, stride=1, dilation=6, padding=6)
        self.conv2 = torch.nn.Conv2d(13, 17, 3, stride=1, dilation=6, padding=6)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t3 = torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5 = t1 * t4
        t6 = t5 / 6
        t7 = self.conv2(t6)
        t8 = t7 * 0.1 + 0.5
        return t8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
