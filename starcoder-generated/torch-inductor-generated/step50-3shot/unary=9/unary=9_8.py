
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp(0, 6)
        v4 = v3.div(6)
        v5 = self.other_conv(v4)
        v6 = v5.add(3)
        return v6
# Inputs to the model
x = torch.randn(8, 3, 64, 64)
