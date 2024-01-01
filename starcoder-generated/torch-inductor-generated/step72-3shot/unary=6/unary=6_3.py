
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 192, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.avgpool(x1)
        t2 = self.conv(t1)
        x1 = torch.clamp_max(t2, 6)
        t3 = x1 + 3
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t1 * t5
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
