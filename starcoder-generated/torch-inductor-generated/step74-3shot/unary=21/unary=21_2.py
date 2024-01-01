
class ModelTanh(torch.nn.Module):
    def __init__(self, num_classes=64):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 12, 5, stride=2, padding=3, dilation=1, groups=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
input = torch.randn(1, 1, 100, 200)
model = ModelTanh()
