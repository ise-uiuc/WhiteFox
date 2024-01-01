
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 8, 3, stride=3, padding=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(2)
        v3 = v2.clamp(min=0, max=6)
        v4 = v3.div(6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 256, 256)
