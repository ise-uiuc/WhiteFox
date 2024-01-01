
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1, track_running_stats=False, affine=False, momentum=0.9)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = self.bn
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
