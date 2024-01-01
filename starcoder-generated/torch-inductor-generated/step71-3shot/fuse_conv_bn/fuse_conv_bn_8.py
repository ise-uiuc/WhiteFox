
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1, groups=1)
        self.bn1 = torch.nn.BatchNorm2d(2, affine=False)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, groups=1)
        self.bn2 = torch.nn.BatchNorm2d(2, affine=False, track_running_stats=True)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return y
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
