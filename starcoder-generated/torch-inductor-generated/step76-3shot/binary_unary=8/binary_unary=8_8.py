
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.conv1(x1)
        v4 = v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
