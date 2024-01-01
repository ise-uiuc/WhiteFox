
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_max(v1, 6)
        v3 = 3 + v2
        v4 = self.bn(v3)
        t1 = self.relu6(v4)
        v5 = v1 * t1
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 512, 512)
