
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_max(v1, 8)
        v3 = 7 + v2
        v4 = 2 - v3
        v5 = torch.clamp_min(v4, -6) 
        v6 = 7 - v5
        v7 = v6 * v1
        v8 = v7 / 6
        v9 = self.relu6(v8)
        v10 = self.bn(v9)
        return v10
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
