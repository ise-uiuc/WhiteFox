
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=0)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 6))
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1 + 3
        t4 = torch.clamp_min(t2, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = self.avgpool(x1).squeeze(-1)
        t7 = t6 * t5
        t8 = t7 / 6
        return t8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
