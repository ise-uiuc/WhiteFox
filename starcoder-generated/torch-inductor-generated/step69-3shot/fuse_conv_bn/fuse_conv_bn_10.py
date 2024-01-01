
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 64, 5)
        self.bn = torch.nn.BatchNorm2d(64, track_running_stats=True)
    def forward(self, x):
        o1 = torch.nn.functional.relu(self.conv1(x))
        o2 = torch.nn.functional.relu(self.conv2(o1))
        o3 = self.bn(o2)
        return o3
# Inputs to the model
x = torch.randn(1, 3, 56, 56)
