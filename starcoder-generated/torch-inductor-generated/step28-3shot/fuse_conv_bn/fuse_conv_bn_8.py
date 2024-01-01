
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 3, 2)
        self.bn1 = torch.nn.BatchNorm2d(2, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(2, 1, 1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.bn1(x2)
        x4 = self.conv2(x3)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)

