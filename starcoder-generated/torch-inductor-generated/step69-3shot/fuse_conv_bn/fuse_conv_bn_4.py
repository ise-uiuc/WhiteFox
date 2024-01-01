
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(6)
        self.conv = torch.nn.Conv2d(3, 4, 3, bias=False)
        self.bn = torch.nn.BatchNorm2d(4, track_running_stats=True) # track_running_stats must be set to True
    def forward(self, x):
        return self.bn(self.conv(x))
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
