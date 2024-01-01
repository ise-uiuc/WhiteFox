
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(33, 33, 3, stride=1, padding=1, groups=33)
        self.bn = torch.nn.BatchNorm2d(33)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.bn(t1)
        t3 = self.avgpool(t2)
        t4 = self.conv1(t3)
        t5 = 3 + t4
        t6 = torch.clamp_min(t5, 0)
        t7 = torch.clamp_max(t6, 6)
        t8 = t4 * t7
        t9 = t8 / 6
        return t4.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(3, 33, 30, 30)
