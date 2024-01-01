
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = v0.add(3)
        v2 = v1.clamp(0, 6)
        v3 = v2.div(6)
        v4 = self.conv(x1)
        v4 = v4.add(3)
        v5 = v4.clamp(0, 6)
        v6 = v5.div(6)
        return v6
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
