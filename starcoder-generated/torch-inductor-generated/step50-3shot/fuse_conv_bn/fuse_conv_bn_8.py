
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(20, 20, 5)
        self.bn = torch.nn.BatchNorm2d(20, track_running_stats=False)
    def forward(self, x):
        return self.bn(self.conv(x))
# Inputs to the model
x = torch.randn(1, 20, 5, 5)
