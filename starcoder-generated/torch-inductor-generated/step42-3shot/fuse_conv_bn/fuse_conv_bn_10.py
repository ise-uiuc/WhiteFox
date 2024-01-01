
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, groups=1, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, groups=1, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(1, affine=True, track_running_stats=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        y = self.conv2(x)
        y = self.bn2(y) # fusion is not possible since the last fusion output is 'y'
        z = x * y
        return z
# Inputs to the model
x = torch.randn(1, 1, 6, 6)
