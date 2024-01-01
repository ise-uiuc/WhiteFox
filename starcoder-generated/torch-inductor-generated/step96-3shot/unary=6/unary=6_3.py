
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU6()
    def forward(self, x1):
        r1 = self.conv(x1)
        r2 = 3 + r1
        r3 = torch.clamp_min(r2, 0)
        r4 = torch.clamp_max(r3, 6)
        r5 = r1 * r4
        r6 = r5 / 6
        r7 = self.bn(r6)
        r8 = self.relu(r7)
        return r8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
