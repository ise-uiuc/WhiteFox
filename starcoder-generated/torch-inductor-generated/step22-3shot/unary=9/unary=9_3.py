
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.other_conv = torch.nn.Conv2d(8, 8, 9, stride=3, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.clamp(min=0, max=6)
        v4 = self.other_conv(v2)
        v5 = v4.clamp(min=0, max=6)
        return v5
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
