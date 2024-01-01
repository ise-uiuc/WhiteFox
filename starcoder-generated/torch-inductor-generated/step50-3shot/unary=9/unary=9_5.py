
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.add(3)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2.div(6)
        v4 = self.other_conv(v3)
        v5 = v4.add(3)
        v6 = v5.clamp(min=0, max=6)
        v7 = v6.div(6)
        return v7
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
