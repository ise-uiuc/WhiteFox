
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(1, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.avgpool(x1)
        t2 = self.conv(t1)
        t3 = 6 + t2
        t4 = torch.clamp(t3, 0)
        t5 = t2 * t4
        t6 = t5 / 6
        return t6
# Inputs to the model
x1 = torch.randn(1, 4, 300, 300)
