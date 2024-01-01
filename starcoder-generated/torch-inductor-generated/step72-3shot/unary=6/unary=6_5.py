
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3, track_running_stats=False)
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.bn(x1)
        t2 = self.conv(t1)
        t3 = t2 + 3
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
