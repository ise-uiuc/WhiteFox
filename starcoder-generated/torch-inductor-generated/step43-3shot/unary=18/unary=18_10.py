
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(num_features=28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = torch.nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn4 = torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, v1):
        v2 = self.bn1(v1)
        v3 = self.bn2(v2)
        v4 = self.bn3(v3)
        v5 = self.bn4(v4)
        return v5
# Inputs to the model
v1 = torch.randn(1, 64, 34, 298)
