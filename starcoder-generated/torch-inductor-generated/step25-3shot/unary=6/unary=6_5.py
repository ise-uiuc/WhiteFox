
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 25, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(25, 96, 4, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(96)
    def forward(self, x1):
        s1 = self.conv1(x1)
        s2 = self.conv2(s1)
        s3 = s2 + 3
        s4 = torch.clamp_min(s3, 0)
        s5 = torch.clamp_max(s4, 6)
        s6 = s1 * s5
        s7 = s6 / 6
        s8 = self.bn(s7)
        return s7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
