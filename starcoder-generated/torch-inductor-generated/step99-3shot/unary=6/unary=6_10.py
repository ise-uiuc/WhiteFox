
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=11, stride=1, padding=5)
        self.conv = torch.nn.Conv2d(16, 48, 1, stride=1, padding=2)
    def forward(self, x1):
        t1 = self.avgpool(x1)
        t2 = self.conv(t1)
        t3 = 3 + t2
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(1, 16, 224, 224)
