
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 5, 2, stride=3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(5, track_running_stats=False)
    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
