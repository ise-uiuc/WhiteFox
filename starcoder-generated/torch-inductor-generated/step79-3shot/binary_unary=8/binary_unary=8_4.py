
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(128, 512, 1, stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        t1 = self.conv1(x1)
        v1 = self.bn(t1)
        v5 = self.bn2(self.conv2(v1))
        t2 = v5 + v1
        v2 = torch.relu(t2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
