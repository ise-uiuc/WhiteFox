
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, padding=1, dilation=2)
        self.bn = torch.nn.BatchNorm2d(6, track_running_stats=False) 
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
