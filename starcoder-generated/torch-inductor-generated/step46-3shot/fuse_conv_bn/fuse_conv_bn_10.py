
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(16, 8, 3, dilation=3, stride=2, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm1d(8)
        self.bn.track_running_stats = False
    def forward(self, x):
        return self.conv(x)
# Inputs to the model
x = torch.randn(1, 16, 16)
