
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        v1 = 3 + self.conv(x1)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2.div(6)
        v4 = self.other_conv(v3)
        v5 = v4.clamp(min=0, max=6)
        v6 = v5.div(6)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
