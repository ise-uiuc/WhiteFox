
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, 1, 2)
        self.bn = torch.nn.BatchNorm2d(4, track_running_stats=False)
        self.pool = torch.nn.MaxPool2d(2)
    def forward(self, x0):
        x0 = self.conv(x0)
        x0 = self.bn(x0)
        x0 = self.pool(x0)
        return x0
# Inputs to the model
x0 = torch.randn(1, 2, 5, 6)
