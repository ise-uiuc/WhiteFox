
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, stride=1, padding=0)
        self.bn_1 = torch.nn.BatchNorm2d(4, eps=1e-12, momentum=0.1, affine=True, track_running_stats=True)
        self.bn_2 = torch.nn.BatchNorm2d(4, eps=1e-12, momentum=0.1, affine=False, track_running_stats=True)
        self.bn_3 = torch.nn.BatchNorm2d(4, eps=1e-12, momentum=0.1, affine=True, track_running_stats=False)
        self.bn_4 = torch.nn.BatchNorm2d(4, eps=1e-12, momentum=0.1, affine=False, track_running_stats=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn_1(v1)
        v3 = self.bn_2(v2)
        v4 = self.bn_3(v3)
        v5 = self.bn_4(v4)
        return v5
# Inputs to the model
x1 = torch.randn(10, 3, 224, 224)
