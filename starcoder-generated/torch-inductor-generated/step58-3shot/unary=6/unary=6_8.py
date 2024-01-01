
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(3, affine=False, track_running_stats=True)
        self.bn2 = torch.nn.BatchNorm2d(3, affine=True, track_running_stats=False)
        self.bn3 = torch.nn.BatchNorm2d(3, affine=True, track_running_stats=True)
        self.bn4 = torch.nn.BatchNorm2d(3, affine=False)
        self.bn5 = torch.nn.BatchNorm2d(3, affine=True)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = self.bn2(v1)
        v3 = self.bn3(v2)
        v4 = self.bn4(v3)
        v5 = self.bn5(v4)
        w = v6 + 3
        return w
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
