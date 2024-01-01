
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = v1.sigmoid()
        v2 = v1.mul(v1)
        v3 = self.conv1(v2)
        v3 = v3.sigmoid()
        v4 = v3.mul(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
