
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, dilation=1, groups=16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v1 * v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
