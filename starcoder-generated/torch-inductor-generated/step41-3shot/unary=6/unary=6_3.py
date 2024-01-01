
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1, dilation=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4
        w = tuple(1 for _ in range(t5.dim()))
        s = tuple(2 for _ in range(t5.dim()))
        t7 = F.avg_pool2d(t5.contiguous(), kernel_size=1, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
        t8 = t7.contiguous().view(t7.size(0), -1)
        t9 = self.bn(t8)
        return t9
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
