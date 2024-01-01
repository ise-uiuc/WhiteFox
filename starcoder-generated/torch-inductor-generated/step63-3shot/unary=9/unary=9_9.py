
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = nn.Conv2d(8, 8, 9)
        self.final_conv = nn.Conv2d(8, 1, 9)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.other_conv(v1.add(3).clamp(0, 6).div(6))
        v3 = self.other_conv(v2.mul(0.00390625).div(6))
        v4 = self.final_conv(v3)
        return v4
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
