
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(77)
        self.conv = torch.nn.Conv2d(3, 3, (2), stride=1, padding=2, dilation=(3), groups=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(1, affine=False, track_running_stats=False)
    def forward(self, x1):
        s = self.conv(x1)
        t = self.bn(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
