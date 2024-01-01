
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 896, 3, stride=1, padding=5, groups=128)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 128, 16, 16)
