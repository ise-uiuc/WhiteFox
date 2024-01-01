
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 11, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.bn = torch.nn.BatchNorm2d(1, affine=False, track_running_stats=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(32, 3, 112, 112)
