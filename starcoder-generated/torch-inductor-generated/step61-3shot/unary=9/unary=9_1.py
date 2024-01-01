
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=3)
    def forward(self, x1):
        v1 = 3 * self.conv(x1)
        v2 = 7 * v1
        v3 = v2.clamp(0, 6)
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
