
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 1, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = x2 - v1
        v3 = v2.clamp(0, 6)
        v4 = v3.div(6)
        v5 = self.other_conv(v4)
        v6 = x2 - v5
        v7 = v6.clamp(0, 6)
        v8 = v7.div(6)
        return v8
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
x2 = torch.randn(1, 8, 32, 32)
