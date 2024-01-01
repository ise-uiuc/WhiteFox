
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.linear = torch.nn.Linear(16384, 256, bias=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn = torch.nn.BatchNorm1d(num_features=256, eps=0.00010000000000000001, momentum=0.10000000000000001, affine=True, track_running_stats=True)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = x1.flatten(0, 1)
        x3 = self.linear(x2)
        x4 = self.relu(x3)
        x5 = self.bn(x4)
        return x5
# Inputs to the model
x = torch.randn(1, 64, 1, 1)
