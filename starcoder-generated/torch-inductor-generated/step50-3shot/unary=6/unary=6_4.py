
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv2d(3, 384, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.avgpool = torch.nn.AvgPool2d(2, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.pool(x1)
        t2 = self.relu(t1)
        t3 = self.conv(t2)
        t4 = 3+t3
        t5 = torch.clamp_min(t4, 0)
        t6 = torch.clamp_max(t5, 6)
        t7 = t6/6
        t8 = self.bn(t7)
        t9 = self.avgpool(t8)
        return t9.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(2, 1, 28, 28)
