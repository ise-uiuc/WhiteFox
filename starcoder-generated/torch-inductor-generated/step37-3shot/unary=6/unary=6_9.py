
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v1.mul(v4)
        v6 = v5 / 6
        t1 = self.bn(v6)
        v7 = torch.nn.functional.relu(t1)
        v8 = v7.clamp_min(0)
        v9 = v8.clamp_max(6)
        v10 = v6.mul(v9)
        v11 = v10 / 6
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
