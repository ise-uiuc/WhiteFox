
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, affine=False, track_running_stats=False)
        self.pool = torch.nn.AvgPool2d(2)
    def forward(self, x):
        y2 = self.pool(self.conv(x))
        return y2
# Inputs to the model
x = torch.randn(1, 3, 24, 16)
