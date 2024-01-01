
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 24, 1, stride=1, padding=0)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(24, 32, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = self.conv(v1)
        t1 = self.conv0(x1).unsqueeze(1)
        v3 = v2 + t1
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7.unsqueeze(0)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
