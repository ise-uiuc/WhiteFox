
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(1, eps=1, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flatten(v1.T, 1)
        v3 = self.bn(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 192, 182)
# Inputs end
