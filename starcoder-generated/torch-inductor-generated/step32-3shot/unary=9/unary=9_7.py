
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other_conv = torch.nn.Conv2d(8, 8, 3).add(3).clamp(min=0, max=6).div(6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v3 / 6
        v5 = self.other_conv(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
