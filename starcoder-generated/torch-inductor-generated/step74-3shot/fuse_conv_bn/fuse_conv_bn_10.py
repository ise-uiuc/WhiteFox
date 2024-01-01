
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = torch.nn.Conv2d(100, 100, 1)
        self.bn = torch.nn.BatchNorm2d(100, affine=False, tracking_running_stats=True)
    def forward(self, x):
        x1 = self.conv1x1(x)
        x1 = self.bn(x1)
        x2 = self.conv1x1(x1)
        x3 = self.conv1x1(x1)
        return (x2, x3)
# Inputs to the model
x = torch.randn(2, 100, 5, 5)
