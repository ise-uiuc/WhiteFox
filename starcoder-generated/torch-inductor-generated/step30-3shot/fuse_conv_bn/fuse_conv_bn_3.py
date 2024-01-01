
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        #self.bn = torch.nn.BatchNorm2d(3, track_running_stats=False) # Error: track_running_stats is False
        bn = torch.nn.BatchNorm2d(3)
        bn.track_running_stats = False
        self.relu = torch.nn.ReLU()
        self.bn = bn
    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = self.relu(x1)
        y = self.bn(x2)
        z = self.relu(y)
        return z
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
x2 = torch.randn(1, 3, 6, 6)
