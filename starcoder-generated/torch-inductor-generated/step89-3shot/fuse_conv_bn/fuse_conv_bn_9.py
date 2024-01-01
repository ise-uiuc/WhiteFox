
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(1, track_running_stats=False)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.conv(y)
        y = self.bn(y)
        return y
# Inputs to the model
x = torch.randn(1, 1, 4)
