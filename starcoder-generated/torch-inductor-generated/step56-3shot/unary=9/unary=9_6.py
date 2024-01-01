
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(3, 8, 1, padding=1)
    def forward(self, x1):
        v1 = self.batchnorm(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
