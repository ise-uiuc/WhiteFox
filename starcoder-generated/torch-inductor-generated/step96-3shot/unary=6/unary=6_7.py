
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 5, stride=1, padding=16)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        s1 = self.conv(x1)
        s2 = s1 + 3
        s3 = torch.clamp_min(s2, 0)
        s4 = torch.clamp_max(s3, 6)
        s5 = torch.mul(s1, s4)
        s6 = torch.div(s5, 6.0)
        return s6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
