
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 10, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = t2 + 3
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        t8 = self.bn(t7)
        return t8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
