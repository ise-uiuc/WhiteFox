
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp_(min=0, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        t1 = self.bn(v5)
        v6 = self.relu(t1)
        v7 = torch.clamp_min(v6, 0)
        v8 = torch.clamp_max(v7, 6)
        v9 = v5 * v8
        v10 = v9 / 6
        return v10

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
