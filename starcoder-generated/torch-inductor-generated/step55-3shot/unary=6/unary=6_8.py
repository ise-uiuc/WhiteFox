
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.avgpool = torch.nn.AvgPool2d(5, 3, 1)
    def forward(self, x1):
        t1a = self.conv0(x1)
        t1b = self.conv1(t1a)
        t1c = t1a + t1b
        t2 = self.avgpool(t1c)
        t3 = 3 + t2
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        return t7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
