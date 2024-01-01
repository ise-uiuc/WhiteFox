
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(640, 160, 16, groups=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv(v3)
        v5 = v4.sigmoid()
        v6 = v3 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 640, 64, 64)
