
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pw = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_dw = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv_pw(x1)
        v2 = self.conv_dw(v1)
        v3 = self.bn(v2)
        v4 = torch.clamp_min(v3, 0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
