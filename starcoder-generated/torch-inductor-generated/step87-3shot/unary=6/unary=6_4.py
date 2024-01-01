
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(inplace=False)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        t1 = self.relu(v3)
        v4 = self.bn(t1)
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
