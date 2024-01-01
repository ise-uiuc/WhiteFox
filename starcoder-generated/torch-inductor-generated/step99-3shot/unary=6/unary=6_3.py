
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t4 = 3 + t3
        t5 = torch.clamp_min(t4, 0)
        t6 = torch.clamp_max(t5, 6)
        t7 = t3 * t6
        t8 = t7 / 6
        t9 = self.bn(t8)
        return t9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
