
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, affine=True, track_running_stats=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x * 2
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
