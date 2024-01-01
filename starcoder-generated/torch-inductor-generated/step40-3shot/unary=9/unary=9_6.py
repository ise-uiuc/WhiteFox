
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.other_conv(v1 + 3)
        v3 = self.other_conv(v2.clamp_min(0))
        v4 = self.other_conv(v3.clamp_max(6))
        v5 = v4.div(6)
        return v5
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
