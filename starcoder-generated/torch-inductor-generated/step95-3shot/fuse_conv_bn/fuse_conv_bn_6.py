
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2, groups=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(1, affine=False, track_running_stats=False)
    # Note that the bias is required so that the model contains a conv and bn node.
    def forward(self, x7):
        x7 = self.conv(x7)
        x7 = self.bn(x7)
        return x7
# Inputs to the model
x7 = torch.randn(1, 1, 32, 64)
