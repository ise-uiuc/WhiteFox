
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.pad(v1, (1,1,1,1))
        v3 = self.bn(v2)
        v4 = 3 + v3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = self.relu6(v6)
        return v7
# Inputs to the model
x1 = torch.randn(10, 3, 224, 224)
