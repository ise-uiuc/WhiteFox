
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dwsconv = torch.nn.Conv2d(2048, 1024, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(1024)
        self.conv = torch.nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.dwsconv(x1)
        t2 = self.bn(t1)
        t3 = self.conv(t2)
        t4 = 3 + t3
        t5 = torch.clamp_min(t4, 0)
        t6 = torch.clamp_max(t5, 6)
        t7 = t3 * t6
        t8 = t7 / 6
        return t8.transpose(-1, -2).contiguous()
# Inputs to the model
x1 = torch.randn(1, 2048, 1, 1)
