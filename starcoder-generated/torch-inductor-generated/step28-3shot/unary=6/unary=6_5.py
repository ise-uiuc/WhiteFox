
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(11, 22, 10, stride=4, padding=2)
        self.bn = torch.nn.BatchNorm2d(22)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.exp(v1)
        v3 = torch.clamp_min(v2, 0.0)
        v4 = v3 / 100
        v5 = self.bn(v4)
        v6 = torch.log(v5)
        return v6
# Inputs to the model
x1 = torch.randn(2, 11, 224, 224)
