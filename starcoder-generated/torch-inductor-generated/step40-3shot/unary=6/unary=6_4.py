
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.avgpool(x1)
        t2 = self.conv(t1)
        t3 = 3_torch.clamp_min(t2, 0)
        t4 = torch.clamp_max(t3, 6)
        t5_1 = t2 * t4
        t6_1 = t5_1 / 6
        return t6_1
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
