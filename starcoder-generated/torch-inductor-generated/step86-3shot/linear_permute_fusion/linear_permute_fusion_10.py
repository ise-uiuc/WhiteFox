
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(256, track_running_stats=True)
    def forward(self, x):
        return self.bn(x).clamp_max(5)
# Inputs to the model
x = torch.randn(1, 256, 2, 3)
