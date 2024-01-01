
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(2, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn1(x1)
        y2 = self.bn2(x1)
        return y2
# Inputs to the model used for test
x1 = torch.randn(1, 3, 2, 2)
