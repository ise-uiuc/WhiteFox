
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(24, 48, 1, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.bn(v1)
        v4 = self.bn(v3)
        v5 = v2 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 24, 63, 63)
